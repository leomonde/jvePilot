import math

from cereal import car
from common.conversions import Conversions as CV
from openpilot.common.numpy_fast import clip
from openpilot.selfdrive.car.chrysler.long_carcontroller import LongCarController
from openpilot.selfdrive.car.chrysler.interface import CarInterface

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

# adjust for hills and towing
TORQ_ADJUST_THRESHOLD = 0.3
START_ADJUST_ACCEL_FRAMES = 100
ADJUST_ACCEL_COOLDOWN_MAX = 1
CAN_DOWNSHIFT_ACCEL_FRAMES = 200
TORQ_BRAKE_MAX = -0.1

COAST_WINDOW = CV.MPH_TO_MS * 2
LOW_WINDOW = CV.MPH_TO_MS * 5

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

def create_acc_commands(packer, counter_offset, das_3, enabled, long_active, gas, max_gear, brakes, brake_prep, starting, stopping):
  commands = []

  das_3_values = das_3.copy()
  das_3_values['ACC_AVAILABLE'] = 1
  das_3_values['ACC_ACTIVE'] = enabled
  das_3_values['COUNTER'] = (das_3['COUNTER'] + counter_offset) % 0x10
  das_3_values['ACC_STANDSTILL'] = stopping and long_active
  das_3_values['ACC_GO'] = starting and long_active

  if brakes is not None and long_active:
    das_3_values['ACC_DECEL_REQ'] = long_active
    das_3_values['ACC_DECEL'] = brakes
    das_3_values['ACC_BRK_PREP'] = brake_prep

  if gas is not None and long_active:
    das_3_values['ENGINE_TORQUE_REQUEST_MAX'] = long_active
    das_3_values['ENGINE_TORQUE_REQUEST'] = gas

  if max_gear is not None:
    das_3_values['GR_MAX_REQ'] = max_gear

  commands.append(packer.make_can_msg("DAS_3", 0, das_3_values))

  return commands

class LongCarControllerV2(LongCarController):
  def __init__(self, CP, params, packer):
    super().__init__(CP, params, packer)

    self.last_gas = 0
    self.under_accel_frame_count = 0
    self.torq_adjust = 0.
    self.max_gear = None

  def acc(self, longitudinalPlan, frame, CC, CS, can_sends):
    counter_changed = CS.das_3['COUNTER'] != self.last_das_3_counter
    self.last_das_3_counter = CS.das_3['COUNTER']

    if not CC.enabled or not CS.longControl:
      self.last_gas = 0
      self.under_accel_frame_count = 0
      self.torq_adjust = 0.
      self.max_gear = None
      return None

    # longitudinal
    if self.CP.openpilotLongitudinalControl:
      accel = clip(CC.actuators.accel, self.params.ACCEL_MIN, CarInterface.accel_max(CS))

      starting = False
      stopping = False
      gas = None
      brakes = None
      if not CC.longActive:
        self.last_gas = max(CS.engine_torque, 0.0)

      if CC.longActive:
        starting = CS.out.vEgo < 0.25 and accel > 0.0 # TODO: use CC.actuators.longControlState == LongCtrlState.starting with disabled startAccel?
        stopping = CS.out.vEgo < 0.25 and accel <= 0.0 # TODO: use CC.actuators.longControlState == LongCtrlState.stopping

        pitch = CC.orientationNED[1] if len(CC.orientationNED) > 1 else 0
        drag_force = calc_drag_force(CS.engine_torque, CS.transmission_gear, pitch, CS.out.aEgo, CS.out.vEgo)
        torque = calc_engine_torque(accel, pitch, CS.transmission_gear, drag_force)
        torque = torque + (self.torq_adjust if torque > 0 else 0)  # adjust for hills and towing
        gas = clip(min(torque, self.last_gas + 2), CS.torqMin + 1, CS.torqMax)
        self.last_gas = gas

        vTarget = longitudinalPlan.speeds[-1]
        speed_to_far_off = abs(CS.out.vEgo - vTarget) > COAST_WINDOW
        engine_brake = TORQ_BRAKE_MAX < accel < 0.0 and not speed_to_far_off and vTarget > LOW_WINDOW and torque > CS.torqMin

        if accel >= 0.0 or engine_brake:
          # adjust for hills and towing
          accel_offset = accel - CS.out.aEgo

          if accel_offset > TORQ_ADJUST_THRESHOLD:
            self.under_accel_frame_count = self.under_accel_frame_count + 1  # inc under accelerating frame count
            if frame - self.under_accel_frame_count > START_ADJUST_ACCEL_FRAMES:
              self.torq_adjust += accel_offset * (CarInterface.ACCEL_MAX / CarInterface.accel_max(CS))

          if self.under_accel_frame_count == 0:
            self.max_gear = None
          elif self.under_accel_frame_count > CAN_DOWNSHIFT_ACCEL_FRAMES:
            if CS.out.vEgo < accel - COAST_WINDOW / CarInterface.accel_max(CS) \
              and CS.out.aEgo < CarInterface.accel_max(CS) / 5 \
              and gas > CS.torqMax * 0.98:  # Time to downshift?
              if CS.transmission_gear > 3 and CS.gasRpm < 4500:
                self.max_gear = CS.transmission_gear - 1
                self.under_accel_frame_count = 0
        else:
          self.max_gear = None
          brakes = min(accel, 0)

        if accel < 0.0:
          self.under_accel_frame_count = 0
          self.torq_adjust = max(0, self.torq_adjust - max(accel * 10, ADJUST_ACCEL_COOLDOWN_MAX))

      brake_prep = brakes is not None and len(longitudinalPlan.accels) and longitudinalPlan.accels[0] - longitudinalPlan.accels[-1] > 1.0
      can_sends.extend(create_acc_commands(self.packer,
                                           2 if counter_changed else 3,
                                           CS.das_3,
                                           CC.enabled,
                                           CC.longActive,
                                           gas,
                                           self.max_gear,
                                           brakes,
                                           brake_prep,
                                           starting,
                                           stopping))
