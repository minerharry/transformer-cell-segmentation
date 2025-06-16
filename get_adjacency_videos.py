import itertools
import operator
import re
from typing import Callable, Iterable, Sequence, TypeVar, overload
from libraries.filenames import filename_regex_anybasename,filename_regex_format



T = TypeVar("T")
@overload
def get_video_groups(filenames:Iterable[T],length:int|None=3,overlap:bool=True,filename_key:Callable[[T],str]=None,filename_regex:str=filename_regex_anybasename,format_regex:str=filename_regex_format,idx_key=2)->list[list[T]]: ...
@overload
def get_video_groups(filenames:Iterable[str],length:int|None=3,overlap:bool=True,filename_key:Callable[[T],str]|None=None,filename_regex:str=filename_regex_anybasename,format_regex:str=filename_regex_format,idx_key=2)->list[list[str]]: ...

def get_video_groups(filenames:Iterable[str]|Iterable[T],length:int|None=3,overlap:bool=True,filename_key:Callable[[T],str]|None=None,filename_regex:str=filename_regex_anybasename,format_regex:str=filename_regex_format,idx_key=2):
    if filename_key is not None:
        names = map(filename_key,filenames)
    else:
        names = filenames
        
    matches = [re.match(filename_regex,fname) for fname in names]    

    matchgroups:list[tuple[str,...]] = [m.groups() for m in matches if m is not None]

    def without_idx(t:tuple[str,...])->tuple[tuple[str,...],tuple[str,...]]:
        return (t[:idx_key], t[idx_key+1:])

    matchgroups.sort(key=without_idx)
    movie_keys:list[list[tuple[str,...]]] = []
    for key,group in itertools.groupby(matchgroups,key=without_idx):
        frames = sorted(list(map(int,map(operator.itemgetter(idx_key),group))))

        if length is None: #use all contiguous frames
            diff = set(range(min(frames),max(frames)+1)) - set(frames)
            assert len(diff) == 0, f"Error: non-contiguous frames in movie; missing frames: {diff}"
            movie_keys.append([key[0] + (str(fr),) + key[1] for fr in frames])
        else:
            while len(frames) != 0: #try to extract contiguous movies from frames
                first = frames.pop(0)
                frs = range(first+1,first+length)
                if all([fr in frames for fr in frs]): #there's a contiguous chunk of the proper length
                    #extract the frames
                    movie = [first]
                    [movie.append(fr) for fr in frs]

                    movie_keys.append([key[0] + (str(fr),) + key[1] for fr in movie])

                    if not overlap: #remove all the other frames we just used too
                        [frames.pop(frames.index(fr)) for fr in frs]
    res = []
    for movie in movie_keys:
        mov = []
        res.append(mov)
        for key in movie: #now get the filename matching the movie key
            reg = re.compile(format_regex.format(*key))
            fname = list(filter(lambda x:
                                reg.match(filename_key(x)) if filename_key is not None else reg.match(x),
                                filenames))
            assert len(fname) == 1, "Improper filename detection using format_regex. Duplicate filenames not captured by the regex?"
            mov.append(fname[0])
    
    return res