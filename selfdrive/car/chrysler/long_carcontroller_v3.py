import math

from cereal import car
from common.conversions import Conversions as CV
from openpilot.selfdrive.car.chrysler import chryslercan
from openpilot.selfdrive.car.chrysler.long_carcontroller import LongCarController
from openpilot.selfdrive.car.chrysler.interface import CarInterface

LongCtrlState = car.CarControl.Actuators.LongControlState

# LONG PARAMS
LOW_WINDOW = CV.MPH_TO_MS * 5
SLOW_WINDOW = CV.MPH_TO_MS * 20
COAST_WINDOW = CV.MPH_TO_MS * 2

# accelerator
ACCEL_TORQ_SLOW = 40  # add this when going SLOW

# ACCEL_TORQ_MAX = 360
UNDER_ACCEL_MULTIPLIER = 1.
TORQ_RELEASE_CHANGE = 0.35
TORQ_ADJUST_THRESHOLD = 0.3
START_ADJUST_ACCEL_FRAMES = 100
CAN_DOWNSHIFT_ACCEL_FRAMES = 200
ADJUST_ACCEL_COOLDOWN_MAX = 1
MIN_TORQ_CHANGE = 2
ACCEL_TO_NM = 1200
TORQ_BRAKE_MAX = -0.1

# braking
BRAKE_CHANGE = 0.06


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

def acc_command(packer, counter_offset, go, gas, max_gear, stop, brake, brake_prep, das_3):
  values = das_3.copy()  # forward what we parsed
  values['ACC_AVAILABLE'] = 1
  values['ACC_ACTIVE'] = 1
  values['COUNTER'] = (das_3['COUNTER'] + counter_offset) % 0x10

  if go is not None:
    values['ACC_GO'] = go

  if stop is not None:
    values['ACC_STANDSTILL'] = stop

  if brake is not None:
    values['ACC_DECEL_REQ'] = 1
    values['ACC_DECEL'] = brake
    values['ACC_BRK_PREP'] = brake_prep

  if gas is not None:
    values['ENGINE_TORQUE_REQUEST_MAX'] = 1
    values['ENGINE_TORQUE_REQUEST'] = gas

  if max_gear is not None:
    values['GR_MAX_REQ'] = max_gear

  return packer.make_can_msg("DAS_3", 0, values)

class LongCarControllerV3(LongCarController):
  def __init__(self, CP, params, packer):
    super().__init__(CP, params, packer)

    # long
    self.accel_steady = 0
    self.last_brake = None
    self.last_torque = 0.
    self.torq_adjust = 0.
    self.under_accel_frame_count = 0
    self.vehicleMass = CP.mass
    self.max_gear = None

  # T = (mass x accel x velocity x 1000)/(.105 x Engine rpm)
  def acc(self, longitudinalPlan, frame, CC, CS, can_sends):
    counter_changed = CS.das_3['COUNTER'] != self.last_das_3_counter
    self.last_das_3_counter = CS.das_3['COUNTER']

    if not CC.enabled or not CS.longControl:
      self.torq_adjust = 0
      self.last_brake = None
      self.last_torque = None
      self.max_gear = None
      return None

    under_accel_frame_count = 0
    aTarget = longitudinalPlan.speeds[8] if len(longitudinalPlan.speeds) > 8 else CC.actuators.accel
    bTarget = CC.actuators.accel
    vTarget = longitudinalPlan.speeds[-1] if len(longitudinalPlan.speeds) else 0
    long_stopping = CC.actuators.longControlState == LongCtrlState.stopping

    override_request = CS.out.gasPressed or CS.out.brakePressed
    fidget_stopped_brake_frame = CS.out.standstill and CS.das_3['COUNTER'] % 2 == 0  # change brake to keep Jeep stopped
    if not override_request:
      stop_req = long_stopping or (CS.out.standstill and bTarget <= 0)
      go_req = not stop_req and CS.out.standstill

      if go_req:
        under_accel_frame_count = self.under_accel_frame_count = START_ADJUST_ACCEL_FRAMES  # ready to add torq
        self.last_brake = None

      currently_braking = self.last_brake is not None
      speed_to_far_off = abs(CS.out.vEgo - vTarget) > COAST_WINDOW
      engine_brake = TORQ_BRAKE_MAX < aTarget < 0 and not speed_to_far_off and vTarget > LOW_WINDOW \
                     and self.torque(CC, CS, aTarget) + self.torq_adjust > CS.torqMin

      if go_req or ((bTarget >= 0 or engine_brake) and not currently_braking):  # gas
        under_accel_frame_count = self.acc_gas(CC, CS, frame, aTarget, vTarget, under_accel_frame_count)

      elif bTarget < 0:  # brake
        self.acc_brake(CS, bTarget, vTarget, speed_to_far_off)

      elif self.last_brake is not None:  # let up on the brake
        self.last_brake += BRAKE_CHANGE
        if self.last_brake >= 0:
          self.last_brake = None

      elif self.last_torque is not None:  # let up on gas
        self.last_torque -= TORQ_RELEASE_CHANGE
        if self.last_torque <= max(0, CS.torqMin):
          self.last_torque = None

      if stop_req:
        brake = self.last_brake = bTarget # + (0.01 if fidget_stopped_brake_frame else 0.0)
        torque = self.last_torque = None
      elif go_req:
        brake = self.last_brake = None
        torque = math.floor(self.last_torque * 100) / 100
      elif self.last_brake:
        brake = math.floor(self.last_brake * 100) / 100
        torque = self.last_torque = None
      elif self.last_torque:
        brake = self.last_brake = None
        torque = math.floor(self.last_torque * 100) / 100
      else:  # coasting
        brake = self.last_brake = None
        torque = self.last_torque = None
    else:
      self.last_torque = None
      self.last_brake = None
      self.max_gear = None
      stop_req = None
      brake = None
      go_req = None
      torque = None

    if under_accel_frame_count == 0:
      self.max_gear = None
      if bTarget < 0 and self.torq_adjust > 0:  # we are cooling down
        self.torq_adjust = max(0, self.torq_adjust - max(bTarget * 10, ADJUST_ACCEL_COOLDOWN_MAX))
    elif under_accel_frame_count > CAN_DOWNSHIFT_ACCEL_FRAMES:
      if CS.out.vEgo < vTarget - COAST_WINDOW / CarInterface.accel_max(CS) \
          and CS.out.aEgo < CarInterface.accel_max(CS) / 5 \
          and torque > CS.torqMax * 0.98:  # Time to downshift?
        if CS.transmission_gear > 3 and CS.gasRpm < 4500:
          self.max_gear = CS.transmission_gear - 1
          under_accel_frame_count = 0

    self.under_accel_frame_count = under_accel_frame_count

    can_sends.append(chryslercan.acc_log(self.packer, int(self.torq_adjust), aTarget, bTarget, vTarget))

    brake_prep = brake is not None and len(longitudinalPlan.accels) and longitudinalPlan.accels[0] - longitudinalPlan.accels[-1] > 1.0

    can_sends.append(acc_command(self.packer,
                                 2 if counter_changed else 3,
                                 go_req,
                                 torque,
                                 self.max_gear,
                                 stop_req and not fidget_stopped_brake_frame,
                                 brake,
                                 brake_prep,
                                 CS.das_3))

    if brake is not None:
      return brake
    elif torque is not None:
      accel = 0 if CS.out.vEgo == 0 else (torque * .105 * CS.gasRpm) / (self.vehicleMass * CS.out.vEgo)  # torque back to accel
      return accel
    return 0

  @staticmethod
  def torque(CC, CS, aTarget):
    pitch = CC.orientationNED[1] if len(CC.orientationNED) > 1 else 0
    drag_force = calc_drag_force(CS.engine_torque, CS.transmission_gear, pitch, CS.out.aEgo, CS.out.vEgo)
    return calc_engine_torque(aTarget, pitch, CS.transmission_gear, drag_force)

  def acc_gas(self, CC, CS, frame, aTarget, vTarget, under_accel_frame_count):
    accelerating = aTarget > 0 and vTarget > CS.out.vEgo + SLOW_WINDOW
    if accelerating:
      aSmoothTarget = (aTarget + CS.out.aEgo) / 2
    else:
      aSmoothTarget = aTarget
    cruise = self.torque(CC, CS, aSmoothTarget)

    if aTarget > 0:
      # adjust for hills and towing
      offset = aTarget - CS.out.aEgo
      if offset > TORQ_ADJUST_THRESHOLD:
        under_accel_frame_count = self.under_accel_frame_count + 1  # inc under accelerating frame count
        if frame - self.under_accel_frame_count > START_ADJUST_ACCEL_FRAMES:
          self.torq_adjust += offset * (CarInterface.ACCEL_MAX / CarInterface.accel_max(CS))

    if cruise + self.torq_adjust > CS.torqMax:  # keep the adjustment in check
      self.torq_adjust = max(0, CS.torqMax - cruise)

    torque = cruise + self.torq_adjust
    self.last_torque = max(CS.torqMin + 1, min(CS.torqMax, torque))

    return under_accel_frame_count

  def acc_brake(self, CS, aTarget, vTarget, speed_to_far_off):
    brake_target = max(CarInterface.ACCEL_MIN, round(aTarget, 2))
    if self.last_brake is None:
      self.last_brake = min(0., brake_target / 2)
    elif self.last_brake < -.2 and CS.out.aEgo < brake_target:  # are we slowing too much?
      self.last_brake += (BRAKE_CHANGE / 10)
    else:
      tBrake = brake_target
      if not speed_to_far_off and 0 >= tBrake >= -1:  # let up on brake as we approach
        tBrake = (tBrake * 1.1) + .1

      lBrake = self.last_brake
      if tBrake < lBrake:
        diff = min(BRAKE_CHANGE, (lBrake - tBrake) / 2)
        self.last_brake = max(lBrake - diff, tBrake)
      elif tBrake - lBrake > 0.01:  # don't let up unless it's a big enough jump
        diff = min(BRAKE_CHANGE, (tBrake - lBrake) / 2)
        self.last_brake = min(lBrake + diff, tBrake)
