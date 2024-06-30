import math

from cereal import car
from openpilot.common.numpy_fast import clip
from selfdrive.car.chrysler.long_carcontroller import LongCarController

LongCtrlState = car.CarControl.Actuators.LongControlState

TIRE_SIZE = [275, 55, 20] # 275/55R20
# https://x-engineer.org/calculate-wheel-radius/
WHEEL_RADIUS = 0.95 * ((TIRE_SIZE[2] * 25.4 / 2) + (TIRE_SIZE[0] * TIRE_SIZE[1] / 100)) / 1000
AXLE_RATIO = 3.21 # or 3.92
FINAL_DRIVE_RATIOS = [x * AXLE_RATIO for x in [4.71, 3.14, 2.10, 1.67, 1.29, 1.00, 0.84, 0.67]]
# https://web.archive.org/web/20180116135154/https://www.ramtrucks.com/2019/ram-1500.html
CdA = 13.0 / 10.764 # CdA = frontal drag coefficient x area (ft^2 converted to m^2)
# https://www.epa.gov/compliance-and-fuel-economy-data/data-cars-used-testing-fuel-economy
ROLLING_RESISTANCE_COEFF = 46.31 / 5500 # Target Coef A (lbf) / Equivalent Test Weight (lbs.)
VEHICLE_MASS = 2495 # kg
GRAVITY = 9.81 # m/s^2
AIR_DENSITY = 1.225 # kg/m3 (sea level air density of dry air @ 15° C)

def calc_motion_force(aEgo, road_pitch):
  force_parallel = VEHICLE_MASS * aEgo
  force_perpendicular = VEHICLE_MASS * GRAVITY * math.sin(road_pitch)
  return force_parallel + force_perpendicular

def calc_drag_force(engine_torque, transmision_gear, road_pitch, aEgo, vEgo, wind=0):
  if vEgo < 2:
    # https://x-engineer.org/rolling-resistance/
    force_rolling = ROLLING_RESISTANCE_COEFF * VEHICLE_MASS * GRAVITY
    # https://x-engineer.org/aerodynamic-drag/
    force_drag = 0.5 * CdA * AIR_DENSITY * ((vEgo - wind)**2)
    return force_rolling + force_drag

  total_force = engine_torque * FINAL_DRIVE_RATIOS[transmision_gear-1] / WHEEL_RADIUS
  return total_force - calc_motion_force(aEgo, road_pitch)

def calc_engine_torque(accel, pitch, transmission_gear, drag_force):
  force_total = calc_motion_force(accel, pitch) + drag_force
  # https://x-engineer.org/calculate-wheel-torque-engine/
  wheel_torque = force_total * WHEEL_RADIUS
  engine_torque = wheel_torque / FINAL_DRIVE_RATIOS[int(transmission_gear)-1]
  return engine_torque

def create_acc_commands(packer, das_3, enabled, long_active, gas, brakes, starting, stopping):
  commands = []

  das_3_values = {
    'COUNTER': (das_3['COUNTER'] + 1) % 0x10,
    'ACC_AVAILABLE': 1,
    'ACC_ACTIVE': enabled,
    'ACC_DECEL_REQ': brakes < 0.0 if long_active else 0,
    'ACC_DECEL': brakes,
    'ENGINE_TORQUE_REQUEST_MAX': brakes >= 0.0 if long_active else 0,
    'ENGINE_TORQUE_REQUEST': gas,
    'ACC_STANDSTILL': stopping if long_active else 0,
    'ACC_GO': starting if long_active else 0,
    # TODO: does this have any impact on fuel economy when engine braking?
    # TODO: does this disable auto engine start/stop?
    'DISABLE_FUEL_SHUTOFF': 1,
    # TODO: does this have any impact on ACC braking responsiveness?
    # TODO: does this cause a fault if set for too long?
    #'ACC_BRK_PREP': brakes < 0.5 if enabled else 0,
    # TODO: does this have any impact on ACC braking responsiveness?
    #'COLLISION_BRK_PREP': ?,
  }

  commands.append(packer.make_can_msg("DAS_3", 0, das_3_values))

  # das_5_values = {
  #   "FCW_STATE": 0, # off
  #   "FCW_DISTANCE": 0, # off
  # }
  # commands.append(packer.make_can_msg("DAS_5", 0, das_5_values))

  return commands

class LongCarControllerV2(LongCarController):
  def __init__(self, CP, params, packer):
    super().__init__(CP, params, packer)

    self.last_gas = 0

  def acc(self, longitudinalPlan, frame, CC, CS, can_sends):
    if CS.das_3['COUNTER'] == self.last_das_3_counter:
      return None
    self.last_das_3_counter = CS.das_3['COUNTER']

    if not CC.enabled or not CS.longControl:
      self.last_gas = 0
      return None

    # longitudinal
    if self.CP.openpilotLongitudinalControl:
      accel = clip(CC.actuators.accel, self.params.ACCEL_MIN, self.params.ACCEL_MAX)

      starting = False
      stopping = False
      gas = CS.torqMin
      brakes = self.params.INACTIVE_ACCEL
      if not CC.longActive:
        self.last_gas = max(CS.engine_torque, 0.0)

      if CC.longActive:
        starting = CS.out.vEgo < 0.25 and accel > 0.0 # TODO: use CC.actuators.longControlState == LongCtrlState.starting with disabled startAccel?
        stopping = CS.out.vEgo < 0.25 and accel <= 0.0 # TODO: use CC.actuators.longControlState == LongCtrlState.stopping

        pitch = CC.orientationNED[1] if len(CC.orientationNED) > 1 else 0
        drag_force = calc_drag_force(CS.engine_torque, CS.transmission_gear, pitch, CS.out.aEgo, CS.out.vEgo)
        gas = clip(calc_engine_torque(accel, pitch, CS.transmission_gear, drag_force), CS.torqMin, CS.torqMax)
        gas = min(gas, self.last_gas + 2)
        self.last_gas = max(gas, 0)
        # TODO: handle road pitch sometimes causing negative accel to be positive gas (uphill)
        if accel < 0.0:
          gas = 0.0
          brakes = min(accel, 0)

      can_sends.extend(create_acc_commands(self.packer, CS.das_3, CC.enabled, CC.longActive, gas, brakes, starting, stopping))
