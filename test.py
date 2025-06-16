from abc import ABC, abstractmethod


class AB(ABC):
    @abstractmethod
    def do_thing(self,i:int)->str: ...

class C(AB):
    @classmethod
    def do_thing(cls,i:int)->str: 
        return str(i)
    
