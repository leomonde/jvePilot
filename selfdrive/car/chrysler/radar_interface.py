#!/usr/bin/env python3
from opendbc.can.parser import CANParser
from cereal import car
from openpilot.selfdrive.car.interfaces import RadarInterfaceBase
from openpilot.selfdrive.car.chrysler.values import DBC
from common.params import Params
from common.cached_params import CachedParams
from math import tan

RADAR_MSGS_C = list(range(0x2c2, 0x2d4+2, 2))  # c_ messages 706,...,724
RADAR_MSGS_D = list(range(0x2a2, 0x2b4+2, 2))  # d_ messages
LAST_MSG = max(RADAR_MSGS_C + RADAR_MSGS_D)
NUMBER_MSGS = len(RADAR_MSGS_C) + len(RADAR_MSGS_D)

def _create_radar_can_parser(car_fingerprint):
  dbc = DBC[car_fingerprint]['radar']
  if dbc is None:
    return None

  msg_n = len(RADAR_MSGS_C)
  # list of [(signal name, message name or number), (...)]
  # [('RADAR_STATE', 1024),
  #  ('LONG_DIST', 1072),
  #  ('LONG_DIST', 1073),
  #  ('LONG_DIST', 1074),
  #  ('LONG_DIST', 1075),

  messages = list(zip(RADAR_MSGS_C +
                      RADAR_MSGS_D,
                      [25] * msg_n +  # 25Hz (0.04s)
                      [25] * msg_n, strict=True))  # 25Hz (0.04s)

  return CANParser(DBC[car_fingerprint]['radar'], messages, 1)

def _address_to_track(address):
  if address in RADAR_MSGS_C:
    return (address - RADAR_MSGS_C[0]) // 2
  if address in RADAR_MSGS_D:
    return (address - RADAR_MSGS_D[0]) // 2
  raise ValueError("radar received unexpected address %d" % address)

class RadarInterface(RadarInterfaceBase):
  def __init__(self, CP):
    super().__init__(CP)
    self.CP = CP
    self.rcp = _create_radar_can_parser(CP.carFingerprint)
    self.updated_messages = set()
    self.trigger_msg = LAST_MSG

    self.cachedParams = CachedParams()
    self.yRel_multiplier = -1 if Params().get_bool("jvePilot.settings.reverseRadar") else 1

  def update(self, can_strings):
    if self.rcp is None or self.CP.radarUnavailable or self.cachedParams.get_bool("jvePilot.settings.visionOnly", 1000):
      return super().update(None)

    vls = self.rcp.update_strings(can_strings)
    self.updated_messages.update(vls)

    if self.trigger_msg not in self.updated_messages:
      return None

    ret = car.RadarData.new_message()
    errors = []
    if not self.rcp.can_valid:
      errors.append("canError")
    ret.errors = errors

    for ii in self.updated_messages:  # ii should be the message ID as a number
      cpt = self.rcp.vl[ii]
      trackId = _address_to_track(ii)

      if trackId not in self.pts:
        self.pts[trackId] = car.RadarData.RadarPoint.new_message()
        self.pts[trackId].trackId = trackId
        self.pts[trackId].aRel = float('nan')
        self.pts[trackId].yvRel = float('nan')

      if 'LONG_DIST' in cpt:  # c_* message
        self.pts[trackId].dRel = cpt['LONG_DIST']  # from front of car
        self.pts[trackId].yRel = cpt['LAT_ANGLE']  # in car frame's y axis, left is positive
        self.pts[trackId].yRel = tan(self.pts[trackId].yRel) * self.pts[trackId].dRel * self.yRel_multiplier
      else:  # d_* message
        self.pts[trackId].vRel = cpt['REL_SPEED']
        self.pts[trackId].measured = bool(cpt['MEASURED'])

    # We want a list, not a dictionary. Filter out LONG_DIST==0 because that means it's not valid.
    ret.points = [x for x in self.pts.values() if x.measured and 290 > x.dRel > 260]

    self.updated_messages.clear()
    return ret
