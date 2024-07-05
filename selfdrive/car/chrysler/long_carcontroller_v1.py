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

class LongCarControllerV1(LongCarController):
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
    aTarget = longitudinalPlan.speeds[5] if len(longitudinalPlan.speeds) > 5 else 0
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
                     and self.torque(CS, aTarget, vTarget) + self.torq_adjust > CS.torqMin

      if go_req or ((aTarget >= 0 or engine_brake) and not currently_braking):  # gas
        under_accel_frame_count = self.acc_gas(CS, frame, aTarget, vTarget, under_accel_frame_count)

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
        brake = self.last_brake = bTarget
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

  def torque(self, CS, aTarget, vTarget):
    return (self.vehicleMass * aTarget * vTarget) / (.105 * CS.gasRpm)

  def acc_gas(self, CS, frame, aTarget, vTarget, under_accel_frame_count):
    if CS.out.vEgo < SLOW_WINDOW:
      cruise = (self.vehicleMass * aTarget * vTarget) / (.105 * CS.gasRpm)
      cruise += ACCEL_TORQ_SLOW * (1 - (CS.out.vEgo / SLOW_WINDOW))
    else:
      accelerating = aTarget > 0 and vTarget > CS.out.vEgo + SLOW_WINDOW
      if accelerating:
        vSmoothTarget = vTarget
        aSmoothTarget = (aTarget + CS.out.aEgo) / 2
      else:
        vSmoothTarget = (vTarget + CS.out.vEgo) / 2
        aSmoothTarget = aTarget

      cruise = (self.vehicleMass * aSmoothTarget * vSmoothTarget) / (.105 * CS.gasRpm)

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
