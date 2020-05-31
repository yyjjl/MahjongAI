# -*- coding: utf-8 -*-

from ..baseline_cnn2 import SessionManager as SessionManagerBase
from .model import CNNModel


class SessionManager(SessionManagerBase):
    def build_model(self):
        return CNNModel(self.options)
