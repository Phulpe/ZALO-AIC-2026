"""Fail closed on outbound Internet connections during inference."""
import os,socket

ATTEMPTS=[]

def install():
    for name in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE','HF_HUB_DISABLE_TELEMETRY','DO_NOT_TRACK']:
        os.environ[name]='1'
    if getattr(socket,'_datamate_offline',False): return
    original=socket.socket
    class LocalSocket(original):
        def connect(self,address):
            if self.family in (socket.AF_INET,socket.AF_INET6):
                ATTEMPTS.append('connect'); raise RuntimeError('Network disabled for DataMate inference')
            return super().connect(address)
        def connect_ex(self,address):
            if self.family in (socket.AF_INET,socket.AF_INET6):
                ATTEMPTS.append('connect_ex'); raise RuntimeError('Network disabled for DataMate inference')
            return super().connect_ex(address)
        def sendto(self,*args,**kwargs):
            if self.family in (socket.AF_INET,socket.AF_INET6):
                ATTEMPTS.append('sendto'); raise RuntimeError('Network disabled for DataMate inference')
            return super().sendto(*args,**kwargs)
    socket.socket=LocalSocket
    def denied_dns(*args,**kwargs):
        ATTEMPTS.append('dns'); raise RuntimeError('DNS disabled for DataMate inference')
    socket.getaddrinfo=denied_dns
    socket.gethostbyname=denied_dns
    socket.gethostbyname_ex=denied_dns
    socket.gethostbyaddr=denied_dns
    socket._datamate_offline=True
