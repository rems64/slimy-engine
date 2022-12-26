import math
import os
import random
import sys
import time
import colorit
from typing import List, Tuple, Union, Literal
from collections import deque
import numpy as np
import json
from pathlib import Path

import pygame
from sortedcontainers import SortedList, SortedDict

vec2 = pygame.math.Vector2
vec3 = pygame.math.Vector3
vec  = vec2|vec3

sqrt2 = math.sqrt(2)

def random_vec3_in_sphere(origin : vec3, radius : float) -> vec3:
    dir = vec3(2*random.random()-1, 2*random.random()-1, 2*random.random()-1)/sqrt2
    return origin + radius*dir

def random_vec3_at_height(origin : vec3, radius : float) -> vec3:
    dir = vec3(2*random.random()-1, 2*random.random()-1, 0).normalize()*random.random()
    return origin + radius*dir

def perpendicular_vector(v:vec3):
    if v[1] == 0 and v[2] == 0:
        if v[0] == 0:
            raise ValueError('Zero vector')
        else:
            return v.cross(vec3(0, 1, 0)).normalize()
    return v.cross(vec3(1, 0, 0)).normalize()

def random_vec3_in_cone(direction : vec3, angle : float) -> vec3:
    side = np.arctan(angle)
    v = vec3((2*random.random()-1)*side, (2*random.random()-1)*side, 0)
    p1 = perpendicular_vector(direction)
    p2 = direction.cross(p1)
    return direction+v[0]*p1+v[1]*p2

def random_vec2_in_cone(direction : vec2, angle : float) -> vec2:
    r, theta = direction.as_polar()
    v=vec2()
    v.from_polar([r, theta+(2*random.random()-1)*angle])
    return v

def randrange(a:float=0.9, b:float=1.1) -> float:
    return random.random()*(b-a)+a

def set_x(vector : vec3, x : float) -> vec3:
    return vec3(x, vector.y, vector.z)

def set_y(vector : vec3, y : float) -> vec3:
    return vec3(vector.x, y, vector.z)

def set_z(vector : vec3, z : float) -> vec3:
    return vec3(vector.x, vector.y, z)

def is_null_vec2(v : vec2) -> bool:
    return v.x==0 and v.y==0
def clamp(x, a, b):
    return max(a, min(x, b))

def replace_extension(path:str, ext:str)->str:
    p = Path(path)
    p=p.with_suffix('.'+ext)
    return p.resolve().as_posix()

def color_from_vec3(vector:vec3) -> pygame.Color:
    return pygame.Color(int(vector.x), int(vector.y), int(vector.z))


def generate_radial_gradient(color1:vec3, alpha1:int, color2:vec3, alpha2:int, size:vec2=vec2(512, 512)):
    temp_surface = pygame.Surface(size, pygame.SRCALPHA)

    # use this to set the amount of 'segments' we rotate our blend into
    # this helps stop blends from looking 'boxy' or like a cross.
    circular_smoothness_steps = 5

    colour_1 = pygame.Color((int(color1.x), int(color1.y), int(color1.z), alpha1))
    colour_1.r = colour_1.r//circular_smoothness_steps
    colour_1.g = colour_1.g//circular_smoothness_steps
    colour_1.b = colour_1.b//circular_smoothness_steps
    colour_1.a = colour_1.a//circular_smoothness_steps

    colour_2 = pygame.Color((int(color2.x), int(color2.y), int(color2.z), alpha2))
    colour_2.r = colour_2.r//circular_smoothness_steps
    colour_2.g = colour_2.g//circular_smoothness_steps
    colour_2.b = colour_2.b//circular_smoothness_steps
    colour_2.a = colour_2.a//circular_smoothness_steps


    # 3x3 - starter
    radial_grad_starter = pygame.Surface((3, 3), pygame.SRCALPHA)
    radial_grad_starter.fill(colour_1)
    radial_grad_starter.fill(colour_2, pygame.Rect(1, 1, 1, 1))

    # 5x5 - starter
    # radial_grad_starter = pygame.Surface((5, 5))
    # radial_grad_starter.fill((0, 0, 0))
    # radial_grad_starter.fill((255//circular_smoothness_steps,
    #                           255//circular_smoothness_steps,
    #                           255//circular_smoothness_steps), pygame.Rect(2, 1, 1, 3))
    # radial_grad_starter.fill((255//circular_smoothness_steps,
    #                           255//circular_smoothness_steps,
    #                           255//circular_smoothness_steps), pygame.Rect(1, 2, 3, 1))


    radial_grad = pygame.transform.smoothscale(radial_grad_starter, size)

    for i in range(0, circular_smoothness_steps):
        radial_grad_rot = pygame.transform.rotate(radial_grad, (360.0/circular_smoothness_steps) * i)

        pos_rect = pygame.Rect((0, 0), size)

        area_rect = pygame.Rect(0, 0, size.x, size.y)
        area_rect.center = radial_grad_rot.get_width()//2, radial_grad_rot.get_height()//2
        temp_surface.blit(radial_grad_rot, pos_rect,
                        area=area_rect,
                        special_flags=pygame.BLEND_RGBA_ADD)

    # final_pos_rect = pygame.Rect((0, 0), (512, 512))
    # final_pos_rect.center = 400, 400
    return temp_surface


def dt_to_seconds(dt:float) -> int:
    return int(dt*1E6)

class MutableBool:
    def __init__(self, val:bool) -> None:
        self._val = val
        pass

    def get(self):
        return self._val
    
    def set(self, val):
        self._val=val

class BoundingBox:
    def __init__(self, begin:None|vec3=None, end:None|vec3=None) -> None:
        self._begin = begin if begin else vec3()
        self._end = end if end else vec3()
    
    def intersect(self, other:'BoundingBox') -> bool:
        if self._begin.x > other._end.x or self._end.x < other._begin.x:
            return False
        if self._begin.y > other._end.y or self._end.y < other._begin.y:
            return False
        if self._begin.z > other._end.z or self._end.z < other._begin.z:
            return False
        return True

class Timeline:
    def __init__(self) -> None:
        pass

    def get(self, t:float):
        pass

class FloatTimelineConstant(Timeline):
    def __init__(self, val:float) -> None:
        Timeline.__init__(self)
        self._val = val
    
    def get(self, t:float)->float:
        return self._val

class FloatTimelineFadeIn(Timeline):
    def __init__(self, percentage:float) -> None:
        Timeline.__init__(self)
        self._percentage:float=percentage
    
    def get(self, t:float)->float:
        if t>self._percentage:
            return 1.
        return t/self._percentage

class FloatTimelineFadeOut(Timeline):
    def __init__(self, percentage:float) -> None:
        Timeline.__init__(self)
        self._percentage:float=percentage
    
    def get(self, t:float)->float:
        if t<self._percentage:
            return 1.
        return 1-(t-self._percentage)/(1-self._percentage)

class FloatTimelineFadeInOut(Timeline):
    def __init__(self, percentage:float) -> None:
        Timeline.__init__(self)
        self._percentage:float=percentage
    
    def get(self, t:float)->float:
        if t<self._percentage:
            return t/self._percentage
        elif t>1-self._percentage:
            return 1-(t-(1-self._percentage))/(self._percentage)
        else:
            return 1.

class Colors:
    Color = pygame.Color
    black       = (0, 0, 0)
    white       = (255, 255, 255)
    green       = (0, 255, 0)
    red         = (255, 0, 0)
    darkgreen   = (150, 215, 140)

    white_a     = (255, 255, 255, 255)

class logTypes:
    """
    Log types
    """
    info = colorit.Colors.green
    timer = colorit.Colors.blue
    warning = colorit.Colors.yellow
    error = colorit.Colors.red
    trace = colorit.Colors.white

def log(msg, type:Tuple[int, int, int]=logTypes.info) -> None:
    """
    Log the data
    """
    pretext = "[WARN]" if type==logTypes.warning else "[INFO]" if type==logTypes.info or type==logTypes.trace else "[TIME]" if type==logTypes.timer else "[ERRO]"
    print(colorit.color(pretext+" {}".format(msg), type))

def log_newline() -> None:
    """
    Log a new line
    """
    print("")

def logf(frame: int, target_frame: int, *args, **kwargs) -> None:
    """
    Log if target_frame matches frame
    """
    if frame==target_frame:
        log(*args, **kwargs)


class Math:
    @staticmethod
    def lerp(a, b, k):
        return k*(b-a)+a
    
    @staticmethod
    def lerp_squared(a, b, k) -> vec:
        c:vec = b - a
        return k * c * c.length() + a

class Line2d:
    def __init__(self, point:vec2, dir:vec2) -> None:
        self._pt  = point
        self._dir = dir
    
    def is_dot(self) -> bool:
        return is_null_vec2(self._dir)

class Rect2d:
    def __init__(self, top_left: vec2, bottom_right: vec2) -> None:
        self._tl = top_left
        self._br = bottom_right
    
    @property
    def width(self):
        return abs(self._br.x-self._tl.x)

    @property
    def height(self):
        return abs(self._br.y-self._tl.y)
    
    @property
    def size_x(self):
        return abs(self._br.x-self._tl.x)
    
    @property
    def size_y(self):
        return abs(self._br.y-self._tl.y)
    
    def corners(self) -> Tuple[float, float, float, float]:
        xa = min(self._tl.x, self._br.x)
        ya = min(self._tl.y, self._br.y)
        xb = max(self._tl.x, self._br.x)
        yb = max(self._tl.y, self._br.y)
        return xa, ya, xb, yb

    def intersect_box(self, other : 'Rect2d') -> bool:
        xa, ya, xb, yb = self.corners()
        xa_o, ya_o, xb_o, yb_o = other.corners()
        return not (xb<xa_o or yb<ya_o or xa>xb_o or ya>yb_o)
    
    # def intersect_line(self, line : Line2d) -> bool:

        
class Rect3d:
    def __init__(self, begin:vec3|None=None, end:vec3|None=None) -> None:
        self._begin = begin if begin else vec2()
        self._end   = end if end else vec2()

    @property
    def size_x(self):
        return abs(self._end.x-self._begin.x)
    
    @property
    def size_y(self):
        return abs(self._end.y-self._begin.y)

    @property
    def size_z(self):
        return abs(self._end.y-self._begin.y)
        

class Image:
    def __init__(self, name, size, path:str="", data:pygame.Surface|None=None, flags:int=0):
        self.name:str = name
        self.size:vec2 = size
        self._path:str= path
        self._data:pygame.Surface|None = data
        if not self._data:
            if size:
                self._data = pygame.Surface(size, flags)
    
    def set_data(self, data):
        self._data = data
    
    def get_data(self) -> pygame.Surface:
        if not self._data: raise RuntimeError("Uninitialized texture")
        return self._data
    
    def resize(self, size) -> 'Image':
        assert self._data!=None
        self._data = pygame.transform.scale(self._data, size)
        self.size = size
        return self

    
    @property
    def path(self):
        return self._path
    
    @path.setter
    def path(self, val):
        self._path = val

class Tileset:
    def __init__(self, name:str, path:str, tile_width:int, tile_height:int) -> None:
        self.global_image = Globals.game.load_image("tileset_"+name, path)
        self._tiles:list[Image] = []
        self._name = name
        self._sx = int(tile_width)
        self._sy = int(tile_height)
        self._xr = int(self.global_image.size.x/float(tile_width))
        self._yr = int(self.global_image.size.y/float(tile_height))
        self._start_index:int = 1
        for y in range(0, int(self.global_image.size.y), self._sy):
            for x in range(0, int(self.global_image.size.x), self._sx):
                im = Image("tileset_"+name+str(int(x))+"x"+str(int(y)), vec2(self._sx, self._sy), path)
                im.get_data().blit(self.global_image.get_data(), vec2(0, 0), pygame.rect.Rect(x, y, self._sx, self._sy))
                self._tiles.append(im)
    
    def get_random_tile(self) -> Image:
        return self._tiles[random.randrange(0, len(self._tiles))]

    def get_tile(self, index:int) -> Image:
        return self._tiles[index]
    
    def get_tile_component(self, i:int) -> 'SpriteComponent':
        sc = SpriteComponent(None, vec3(), vec2(self._sx, self._sy))
        sc.sprite = self._tiles[i]
        sc._size_locked = True
        return sc


    def get_random_tile_component(self) -> 'SpriteComponent':
        i = random.randint(0, len(self._tiles))
        return self.get_tile_component(i)
    
    
    def get_tile_size(self) -> Tuple[int, int]:
        return (self._sx, self._sy)

class Tilemap:
    def __init__(self, name:str, tilesets:list[Tileset], size:vec2, tile_size:vec2) -> None:
        self._tilesets = SortedDict({t._start_index:t for t in tilesets})
        self._sx = tile_size.x
        self._sy = tile_size.y
        self._size = size if size else vec2()
        self.map   = np.zeros((int(size[0]), int(size[1])), dtype=int)
        self.image = Image(name, (size[0]*self._sx, size[1]*self._sy))
    
    def get_tile(self, idx:int) -> Image:
        index=idx+1
        i=1
        for j in self._tilesets.keys():
            if index<j:
                if index<=i:
                    raise RuntimeError("Tile does not exist")
                return self._tilesets[i].get_tile(index-i-1)
            i=j
        if index>=i:
            return self._tilesets[i].get_tile(index-i-1)
        log("Tile not found {}".format(index), logTypes.error)
        return self._tilesets[self._tilesets.keys()[0]].get_tile(index)

    def compute(self):
        m, n = self.map.shape
        for i in range(m):
            for j in range(n):
                tile = self.get_tile(self.map[i, j])
                self.image.get_data().blit(tile.get_data(), (j*self._sy, i*self._sx))
    
    def set_random(self):
        n = max(self._tilesets.keys())+self._tilesets[max(self._tilesets.keys())]._start_index - 1
        self.map = np.random.randint(n, size=(int(self._size[0]), int(self._size[1])))
        self.compute()
    
    @property
    def width(self):
        m, n = self.map.shape
        return n
    
    @property
    def height(self):
        m, n = self.map.shape
        return m


def import_tiled_tilemap(name:str, path:str, tileset:Tileset) -> Tilemap:
    data = None
    with open(path) as f:
        data=json.load(f)
    log("Loaded tilemap with {} layer{}".format(len(data["layers"]), "s" if len(data["layers"])>1 else ""))
    layer = data["layers"][0]
    width = int(layer["width"])
    height = int(layer["height"])
    size = vec2(width, height)
    map = np.array(layer["data"])
    map = map.reshape(height, width)
    tm = Tilemap(name, [tileset], size, vec2(data["tilewidth"], data["tileheight"]))
    tm.map = map
    tm.compute()
    return tm


def get_image_size_tuple(size):
    if size==None:
        return None
    elif type(size)==list and len(size)==2:
        return (size[0], size[1])
    elif type(size)==vec2 and len(size)==2:
        return (size[0], size[1])
    elif type(size)==tuple and len(size)==2:
        return size
    raise RuntimeError("Unknown type for image size")

class Game:
    def __init__(self, size:tuple[int, int]=(640, 480)):
        if Globals.game: raise RuntimeError("There can exist only one game")
        Globals.game = self
        self.size = size
        self.title = ""
        self._fonts : dict[str, pygame.font.Font] = {}
        self._clock : pygame.time.Clock = pygame.time.Clock()
        self._delta_time:float = 0.0
        self._alive = True
        self._target_fps = 60
        self._background_color = Colors.black
        self._events = []
        self._images = {}
        self.active_scene : Scene = Scene()

        self._frame_debugs = []
        self.debug_infos:dict[str, str] = {"fps": "0", "deltatime": "0"}
        self._no_debug = False
    
    def init(self, title="Slimy Engine"):
        pygame.init()
        flags = pygame.RESIZABLE | pygame.DOUBLEBUF
        self.screen = pygame.display.set_mode(self.size, flags)
        self.title = title
        pygame.display.set_caption(title)
        self._clock = pygame.time.Clock()
        self.load_font("debug_default", "engine/debug_font.ttf")

        self.load_image("default", "engine/default.png")
        self.load_image("default_shadow", "engine/default_shadow.png")
        self.load_image("default_particle", "engine/default_particle.png")

        return self
    
    def load_font(self, name, path, size=28, force_reload=False):
        if (not self._fonts.get(name)) or force_reload:
            self._fonts[name] = pygame.font.Font(path, size)
            return True
        return False
    
    def resource_path(self, relative_path):
        try:
            # PyInstaller creates a temporary folder and stores path in _MEIPASS
            base_path = sys._MEIPASS  # type: ignore # pylint: disable=no-member
        except Exception:
            base_path = os.path.abspath(".")
        
        return os.path.join(base_path, relative_path)

    def load_image(self, name, path="", size=None, force_reload=False):
        if self._images.get(name):
            if (not force_reload) and self._images[name].get(get_image_size_tuple(size)):
                return self._images[name][get_image_size_tuple(size)]
            else:
                if size:
                    p=self.resource_path(next(iter(self._images[name].values())).path)
                    log("Loading image {} from disk with size ({}, {})".format(name, size[0], size[1]), logTypes.trace)
                    im = pygame.image.load(p).convert_alpha()
                    im = pygame.transform.scale(im, size)
                    s = im.get_size()
                    img = Image(name, vec2(s[0], s[1]), path, im)
                    self._images[name][s] = img
                    return img
                else:
                    return self._images[name][max(self._images[name])]
        if not path:
            raise RuntimeError("Never loaded this resource and no path specified ("+name+")")
        im = pygame.image.load(self.resource_path(path)).convert_alpha()
        log("Loading image {} from disk with {}".format(name, "size {}".format(size) if size else "default size"), logTypes.trace)
        if size:
            im = pygame.transform.scale(im, size)
        s = im.get_size()
        img = Image(name, vec2(s[0], s[1]), path, im)
        self._images[name] = {}
        self._images[name][s] = img
        return img

    def is_alive(self):
        return self._alive
    
    def target_fps(self, fps):
        self._target_fps = fps
        return self
    
    def get_delta_time(self) -> float:
        return self._delta_time
    
    def set_background_color(self, color):
        assert type(color)==tuple
        self._background_color = color
        return self
    
    def on_resize(self, event):
        self.size = (event.dict['size'][0], event.dict['size'][1])
        self.camera.update_screen_size(event.dict['size'])
        self.active_scene.update_screen_size(event.dict['size'])
        pass
    
    def begin_frame(self, dont_clear=False):
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                self._alive = False
            if event.type == pygame.VIDEORESIZE:
                self.on_resize(event)
                pygame.display.update()
        if not dont_clear:
            self.screen.fill(self._background_color)
    
    def end_frame(self):
        self._delta_time = self._clock.get_time()
        if not self._no_debug:
            for debug in self._frame_debugs:
                debug.draw(self.screen)
            self._frame_debugs = []
            current_height = 10
            self.debug_infos["fps"] = str(round(self._clock.get_fps()))
            self.debug_infos["deltatime"] = str(round(self._clock.get_time(), 1))
            for debug in self.debug_infos:
                txt = str(debug) + ": " + str(self.debug_infos[debug])
                img = self._fonts["debug_default"].render(txt, True, (255, 255, 255))
                rect = img.get_rect()
                self.screen.blit(img, (self.size[0]-rect.width-10, current_height))
                current_height+=rect.height+5
        
        pygame.display.flip()
        self._clock.tick(self._target_fps)
        return
    
    def draw_debug_vector(self, start : vec3, end : vec3, color=(255,0,0), immediate=False):
        if self._no_debug: return
        vector = DebugVector(self)
        vector.start = start
        vector.end = end
        vector.color = color
        if immediate:
            vector.draw(self.screen)
        else:
            self._frame_debugs.append(vector)
        
    def draw_debug_spring(self, start : vec3, end : vec3, color=(255,0,0), immediate=False):
        if self._no_debug: return
        spring = DebugSpring(self)
        spring._start = start
        spring._end = end
        spring._color = color
        if immediate:
            spring.draw(self.screen)
        else:
            self._frame_debugs.append(spring)
        
    def draw_debug_rectangle(self, start : vec2, end : vec2, color=(0,0,255), immediate=False, thickness=1):
        if self._no_debug: return
        square = DebugRectangle(self)
        square.start = start
        square.end = end
        square.color = color
        square.thickness = thickness
        if immediate:
            square.draw(self.screen)
        else:
            self._frame_debugs.append(square)

    def draw_debug_box(self, start : vec3, end : vec3, color=(0,0,255), immediate=False, thickness=1):
        if self._no_debug: return
        square = DebugBox(self)
        square.start = start
        square.end = end
        square.color = color
        square.thickness = thickness
        if immediate:
            square.draw(self.screen)
        else:
            self._frame_debugs.append(square)
    
    def load_scene(self, scene:'Scene'):
        self.active_scene = scene
        self.active_scene.active_camera.update_screen_size(vec2(self.size[0], self.size[1]))
        return self
    
    def set_debug(self, debug):
        self._no_debug = not debug
        return self
    
    @property
    def camera(self) -> 'Camera':
        return self.active_scene.active_camera

class Object:
    def __init__(self):
        self._delete_me = False

class Component:
    def __init__(self, parent:Union[None,'Component']=None) -> None:
        self._parent   : Component | None  = parent
        self._children : List[Component]   = []
        if parent:
            assert issubclass(type(parent), Component)
            parent.add_child(self)

    def add_child(self, child):
        self._children.append(child)
    
    def attach(self, parent):
        self._parent = parent
    
    def update(self):
        pass
        
    @property
    def children(self):
        return self._children
    
    def any_child(self):
        return len(self._children)>0

class SceneComponent(Component):
    def __init__(self, parent:Union['SceneComponent',None]=None, pos:vec3|None=None):
        Component.__init__(self, parent)
        self._parent:SceneComponent|None = parent
        self._pos:vec3 = pos if pos else vec3()
        self._size:vec3 = vec3(0.5, 0.5, 0.5)
        self._scene_parent:bool = True if issubclass(type(parent), SceneComponent) else False
        self._parent_pos:vec3 = parent.get_world_position() if self._scene_parent else vec3() # type: ignore
        self._valid:bool = False
        self._update_count:int = 0
        self._inherit_parent_position = True
    
    def get_local_position(self):
        return self._pos
    
    def set_local_position(self, val):
        self._pos = val
        self._valid = False
        # for c in self._children:
        #     c.invalidate()
    
    def set_inherit_parent_location(self, val:bool) -> None:
        self._inherit_parent_position = val
    
    def get_world_position(self):
        return self._parent_pos+self._pos

    @property
    def size(self):
        return self._size
    
    @size.setter
    def size(self, s):
        self._size = s
    
    def set_size(self, size:vec3):
        self._size = size
        return self
    
    def get_size(self):
        return self._size
    
    def attach(self, parent:'SceneComponent'):
        Component.attach(self, parent)
        self._parent_pos = parent.get_world_position() if self._inherit_parent_position else vec3()
        self.invalidate()
    
    def invalidate(self):
        self._valid = False

    def update(self):
        # if self._valid: return
        # self._update_count+=1
        if self._parent: self._parent_pos=self._parent.get_world_position() if self._inherit_parent_position else vec3()
        for child in self._children:
            # child._parent_pos = vec3(self._parent_pos+self._pos)
            child.update()
        self._valid = True
        return

class Camera(SceneComponent):
    def __init__(self, parent:SceneComponent|None=None, pos:vec3|None=None) -> None:
        SceneComponent.__init__(self, parent, pos)
        self.game = Globals.game
        self.zoom = 1.
        pass

    def world_to_screen(self, world : vec3) -> vec2:
        return vec2()
    
    def world_to_cam(self, world : vec3) -> vec3:
        return vec3()

    def update_screen_size(self, size : vec2) -> None:
        pass

    def set_zoom(self, zoom : float) -> None:
        pass

    def get_zoom(self) -> float:
        return self.zoom
    
    def world_size2_to_screen(self, dim : vec2) -> vec2:
        return vec2()

class OrthographicCamera(Camera):
    def __init__(self) -> None:
        Camera.__init__(self)
        self.offset = vec2(0.5, 0.5)                    # Camera center is...centered
        self.aspect_ratio = 1
        self.bounds = vec2(20./self.aspect_ratio, 20.)
        self.screen_size = vec2(640, 480)
    
    def update_screen_size(self, size : vec2) -> None:
        self.aspect_ratio = size[1]/size[0]
        self.screen_size = vec2(size[0], size[1])
        self.bounds = vec2(self.bounds[1]/self.aspect_ratio, self.bounds[1])

    def world_to_cam(self, world : vec3) -> vec3:
        p=self.get_world_position()
        return world-vec3(p.x, p.y, 0)

    def world_to_screen(self, world : vec3) -> vec2:
        p = world-self.get_world_position()
        x = ((p.x)/self.bounds.x + self.offset.x)*self.screen_size.x
        y = ((p.y-world.z)/self.bounds.y + self.offset.y)*self.screen_size.y
        return vec2(int(x), int(y))
    
    def set_zoom(self, zoom : float) -> None:
        self.bounds*=1/abs(self.zoom-zoom)
        self.zoom = clamp(zoom, 0, 3)
    
    def world_size2_to_screen(self, dim : vec2) -> vec2:
        x = dim.x*self.screen_size.x/self.bounds.x
        y = dim.y*self.screen_size.y/self.bounds.y
        return vec2(int(x), int(y))


class Drawable():
    def __init__(self):
        pass
    
    def draw(self):
        return

class DrawableComponent(SceneComponent, Drawable):
    def __init__(self, parent=None, pos=vec3()):
        SceneComponent.__init__(self, parent, pos)
        Drawable.__init__(self)
    
    def __le__(self, other):
        return self._pos.y<=other._pos.y
    
    def __lt__(self, other):
        return self._pos.y<other._pos.y


class Light(DrawableComponent):
    def __init__(self, scene:'Scene', parent: Union['SceneComponent', None] = None, pos: vec3 | None = None):
        DrawableComponent.__init__(self, parent, pos)
        self._strength:float = 1.
        self._scene = scene
        self._light_surface:pygame.Surface=pygame.surface.Surface((10, 10))
        self._color:vec3 = vec3(255, 255, 255)
    
    def set_color(self, color:vec3):
        self._color.x, self._color.y, self._color.z = color.x, color.y, color.z
        return self
    
    def draw(self):
        Drawable.draw(self)
    
    def render(self):
        pass

class PointLight(Light):
    def __init__(self, scene:'Scene', parent: Union['SceneComponent', None] = None, pos: vec3 | None = None):
        Light.__init__(self, scene, parent, pos)
        # self._light_surface = pygame.Surface((100, 100))
        # self._light_surface.fill('White')
        self.render()
    
    def draw(self):
        Light.draw(self)
        self._scene.get_light_map().blit(self._light_surface, Globals.game.camera.world_to_screen(self._pos), special_flags=pygame.BLEND_ADD)
    
    def render(self):
        Light.render(self)
        screen_size = Globals.game.camera.world_size2_to_screen(self.size.xy)
        self._light_surface = generate_radial_gradient(vec3(0, 0, 0), 255, self._color, 255, screen_size)
        return self

class ParticleEmitter(Drawable):
    def __init__(self, system:Union[None,'ParticleSystem']=None):
        Drawable.__init__(self)
        self._rate = 1.0
        self._elapsed_time:int = 0
        self._particles:deque[tuple[int, MutableBool, vec3, vec3, list[int], int]]=deque()  # particle: id, is_alive, position, velocity, color, creation_time
        self._sprite:Image=Globals.game.load_image("default_particle").resize((128, 128))
        self._started = False
        self._system = system
        self.draw_size = vec2()
        self._sprite_size = vec2(5., 5.)
        self._size_locked = False
        self._pos = vec3()
        self._track_component:None|SceneComponent = None
        self._alpha_animate=FloatTimelineFadeInOut(0.1)
    
    def track_component(self, component:SceneComponent):
        self._track_component = component
    
    def start(self) -> 'ParticleEmitter':
        self._started = True
        return self
    
    def tick(self, dt) -> None:
        MAX_AGE=5
        if not self._started: return
        self._elapsed_time+=dt
        if self._track_component:
            self._pos = self._track_component.get_world_position()
        
        self.draw_size = Globals.game.camera.world_size2_to_screen(self._sprite_size)
        if self.draw_size!=self._sprite.size:
            log("Resizing particle sprite", logTypes.warning)
            if (not self._size_locked):
                self._sprite = Globals.game.load_image(self._sprite.name, self._sprite.path, self.draw_size)
            else:
                self._sprite.resize(self.draw_size)
             
        found_alive=False
        up_to=0
        i=-1
        for particle in self._particles:
            i+=1
            id, is_alive, position, velocity, color, creation_time = particle
            if not is_alive.get():
                if not found_alive: up_to+=1
                continue
            found_alive=True
            age = self._elapsed_time-creation_time
            normalized_age:float = age/MAX_AGE
            if normalized_age>1: is_alive.set(False)
            color[3]=int(self._alpha_animate.get(normalized_age)*255)
            
            position+=velocity*dt
        if not found_alive: self._particles=deque()
        else:
            for _ in range(up_to): self._particles.popleft()
        Globals.game.debug_infos["particles_count"]=str(len(self._particles))
        
    def draw(self) -> None:
        assert self._system!=None
        screen = Globals.game.screen
        camera = Globals.game.camera
        screen_size = Globals.game.size
        for particle in self._particles:
            id, is_alive, position, _, color, _ = particle
            if not is_alive.get(): continue
            screen_pos = camera.world_to_screen(position+self._system.get_world_position())

            if screen_pos.x+self.draw_size.x<0 or screen_pos.y+self.draw_size.y<0 or screen_pos.x-self.draw_size.x>screen_size[0] or screen_pos.y-self.draw_size.y>screen_size[1]: is_alive.set(False)
            self._sprite.get_data().set_alpha(color[3])
            screen.blit(self._sprite.get_data(), screen_pos)

class FountainEmitter(ParticleEmitter):
    def __init__(self, system: Union[None, 'ParticleSystem'] = None):
        ParticleEmitter.__init__(self, system)
    
    def tick(self, dt) -> None:
        if not self._started: return

        if random.random()>0.95:
            self._particles.appendleft((0, MutableBool(True), self._pos.copy(), set_z(random_vec3_in_cone(vec3(0, -1, 0), np.pi/4*random.random()+np.pi/8)*1, 0), list(Colors.white_a), self._elapsed_time))

        ParticleEmitter.tick(self, dt)


class ParticleSystem(DrawableComponent):
    def __init__(self, parent=None, pos=vec3()):
        print(parent)
        DrawableComponent.__init__(self, parent, pos)
        self._emitters:list[ParticleEmitter]=[]
    
    def start(self) -> 'ParticleSystem':
        for emitter in self._emitters:
            emitter.start()
        return self

    def tick(self, dt) -> 'ParticleSystem':
        for emitter in self._emitters:
            emitter.tick(dt)
        
        return self

    def draw(self) -> 'ParticleSystem':
        DrawableComponent.draw(self)
        for emitter in self._emitters:
            emitter.draw()
        
        return self

class DebugDraw:
    def __init__(self, game) -> None:
        self.game : Game = game
        pass

    def draw(self, screen):
        return

class DebugVector(DebugDraw):
    def __init__(self, game) -> None:
        DebugDraw.__init__(self, game)
        self.start:vec3         = vec3()
        self.end:vec3           = vec3()
        self.color:Colors.Color = Colors.Color(255, 0, 0)
        self.thickness:int      = 1
    
    def draw(self, screen):
        DebugDraw.draw(self, screen)
        if self.start==self.end: return
        s_2d = self.game.camera.world_to_screen(self.start)
        e_2d = self.game.camera.world_to_screen(self.end)
        if (e_2d-s_2d).length()<=0: return
        dir = (e_2d-s_2d).normalize()
        length = (e_2d-s_2d).length()
        pygame.draw.line(screen, self.color, s_2d, e_2d)
        pygame.draw.lines(screen, self.color, False, [length*0.3*(-dir).rotate(20)+e_2d, e_2d, length*0.3*(-dir).rotate(-20)+e_2d])

class DebugRectangle(DebugDraw):
    def __init__(self, game) -> None:
        DebugDraw.__init__(self, game)
        self.start:vec2 = vec2()
        self.end:vec2 = vec2()
        self.color:Colors.Color = Colors.Color(255, 0, 0)
        self.thickness = 1
    
    def draw(self, screen):
        DebugDraw.draw(self, screen)
        if self.start==self.end: return
        pygame.draw.rect(screen, self.color, pygame.Rect(self.start.x, self.start.y, self.end.x-self.start.x, self.end.y-self.start.y), self.thickness)

class DebugBox(DebugDraw):
    def __init__(self, game) -> None:
        DebugDraw.__init__(self, game)
        self.start:vec3 = vec3()
        self.end:vec3 = vec3()
        self.color:Colors.Color = Colors.Color(255, 0, 0)
        self.thickness = 1
    
    def draw(self, screen):
        DebugDraw.draw(self, screen)
        s2d = self.game.camera.world_to_screen(self.start)
        e2d = self.game.camera.world_to_screen(self.end)
        if self.start==self.end: return
        pygame.draw.rect(screen, self.color, pygame.Rect(s2d.x, s2d.y, e2d.x-s2d.x, e2d.y-s2d.y), self.thickness)


class DebugSpring(DebugDraw):
    def __init__(self, game) -> None:
        DebugDraw.__init__(self, game)
        self._start:vec3 = vec3()
        self._end:vec3 = vec3()
        self._color:Colors.Color = Colors.Color(255, 0, 0)
        self._thickness = 1
    
    def draw(self, screen):
        ends_length = 0.5
        num_spires = 15
        width=0.5
        camera = self.game.camera
        if self._start==self._end: return
        unit = (self._end-self._start).normalize()
        length = (self._end-self._start).length()
        ends_length=min(ends_length, length/2)
        side=vec3(0,0,1).cross(unit).normalize() if vec3(0,0,1).dot(unit)==0 else vec3(0,1,0).cross(unit).normalize()
        offset = self._start+unit*ends_length
        pygame.draw.line(screen, self._color, camera.world_to_screen(self._start), camera.world_to_screen(offset), self._thickness)
        stride = (length-2*ends_length)/num_spires
        for i in range(num_spires):
            pygame.draw.lines(screen, self._color, False, [camera.world_to_screen(offset+i*stride*unit), camera.world_to_screen(offset+i*stride*unit+stride/2*unit+(width*side if i%2 else -width*side)), camera.world_to_screen(offset+(i+1)*stride*unit)])
        pygame.draw.line(screen, self._color, camera.world_to_screen(self._end-unit*ends_length), camera.world_to_screen(self._end), self._thickness)

class Scene:
    def __init__(self):
        self._objects : List[SceneComponent] = []
        self.manual_rendering : bool = False
        self.active_camera : Camera = OrthographicCamera() # type: ignore
        self._drawables : SortedList = SortedList()
        self._tilemaps : list[Tilemap] = []
        self._tilesets : list[Tileset] = []
        self._backgrounds : list[SpriteComponent] = []
        self._ambient_light = vec3(1., 1., 1.)*0.5
        self._lights : list[Light] = []
        self._lightmap : pygame.Surface = pygame.surface.Surface(vec2(10, 10))
    
    def add_drawable_rec(self, obj : SceneComponent):
        if issubclass(type(obj), DrawableComponent):
            self._drawables.add(obj)
        if obj.any_child():
            for child in obj.children:
                if issubclass(type(child), SceneComponent):
                    self.add_drawable_rec(child) # type: ignore
    
    def register_component(self, component : SceneComponent):
        self._objects.append(component) # Add only the root component
        self.add_drawable_rec(component)
        return self
    
    def register_light(self, light:Light):
        self._lights.append(light)
        light._scene = self
        return self
    
    def update_screen_size(self, size:vec2):
        for light in self._lights:
                light.render()
    
    def draw(self):
        if not self.manual_rendering:
            for background in self._backgrounds:
                background.draw()
            for obj in self._drawables:
                obj.draw()
    
    def set_ambient_light(self, val:vec3):
        self._ambient_light = val
        return self
    
    def light_pass(self):
        if self._lightmap.get_size()!=Globals.game.size:
            self._lightmap = pygame.surface.Surface(Globals.game.size)
        self._lightmap.fill(color_from_vec3(self._ambient_light*255))
        if not self.manual_rendering:
            for light in self._lights:
                light.draw()
            
            Globals.game.screen.blit(self._lightmap, (0, 0), special_flags=pygame.BLEND_MULT)
    
    def update(self):
        for obj in self._objects:
            obj.update()
    

    def load_map(self, name:str, path:str) -> list['SpriteComponent']:
        sprites = []
        data = None
        with open(path) as f:
            data=json.load(f)
        d_tilesets = data["tilesets"]
        d_layers = data["layers"]
        log("Loading map with:")
        log(" => {} tileset{}".format(len(d_tilesets), "s" if len(d_tilesets)>1 else ""))
        i=0
        tw = data["tilewidth"]
        th = data["tileheight"]
        for tileset in d_tilesets:
            ts = Tileset(name+"_"+str(i), replace_extension(tileset["source"], "png"), int(data["tilewidth"]), int(data["tileheight"]))
            ts._start_index = tileset["firstgid"]
            self._tilesets.append(ts)
            i+=1
        
        for layer in d_layers:
            width = int(layer["width"])
            height = int(layer["height"])
            size = vec2(width, height)
            map = np.array(layer["data"])
            map = map.reshape(height, width)
            tm = Tilemap(name, self._tilesets, size, vec2(tw, th))
            tm.map = map
            tm.compute()
            self._tilemaps.append(tm)
            map_sprite = SpriteComponent(None, vec3(2, 2, 0), vec2(0, 0))
            map_sprite.size = vec3(tm.width*2, tm.height*2, 0)
            map_sprite.sprite = tm.image
            map_sprite._size_locked = True
            sprites.append(map_sprite)
            self._backgrounds.append(map_sprite)
        
        return sprites
    
    def get_light_map(self) -> pygame.Surface:
        return self._lightmap

class Level:
    def __init__(self) -> None:
        pass

class Event:
    def __init__(self):
        self.consumed = False

class Globals:
    game : Game = None  # type: ignore
    world : 'PhysicsWorld' = None # type: ignore

class PhysicsWorld:
    def __init__(self):
        self.objects:List[PhysicsComponent] = []
        self._particle_systems:list[ParticleSystem] = []
        self.limits = [vec3(-math.inf, -math.inf, -math.inf), vec3(math.inf, math.inf, math.inf)]
        self._draw_borders = True
        self.last_tick = time.time_ns()
        self.tmp_tick = time.time_ns()
    
    def set_limits(self, min, max):
        assert type(min)==vec3 and type(max)==vec3
        self.limits = [min, max]
    
    def register_physics_component(self, obj:'PhysicsComponent'):
        assert issubclass(type(obj), PhysicsComponent)
        self.objects.append(obj)
        obj.world = self

    def register_particle_system(self, obj:'ParticleSystem'):
        assert issubclass(type(obj), ParticleSystem)
        self._particle_systems.append(obj)
        # obj.world = self

    def tick(self):
        self.tmp_tick=time.time_ns()
        dt = (self.tmp_tick-self.last_tick)*1.0E-9
        self.last_tick=self.tmp_tick
        if self._draw_borders and self.limits[0].length_squared()<math.inf and self.limits[1].length_squared()<math.inf:
            Globals.game.draw_debug_box(set_z(self.limits[0], 0), set_z(self.limits[1], 0), vec3(255, 0, 0), thickness=2)
        
        for obj in self.objects:
            obj.tick(dt)
        
        for system in self._particle_systems:
            system.tick(dt)
    
    def line_trace(self, origin:vec3, direction:vec3):
        return vec3(origin.x, origin.y, 0)

class Force:
    def __init__(self, value=vec3()):
        self.value = value
    
    def get(self, object:Union['PhysicsComponent',None]=None):
        return self.value
    
    @property
    def x(self):
        return self.value.x

    @x.setter
    def x(self, val):
        self.value.x=val
    
    @property
    def y(self):
        return self.value.y
    
    @y.setter
    def y(self, val):
        self.value.y=val

    @property
    def z(self):
        return self.value.z
    
    @z.setter
    def z(self, val):
        self.value.z=val

    @property
    def length(self):
        return self.value.length()
    
    def normalize_ip(self):
        self.value.normalize_ip()
    
    def scale(self, val):
        self.value*=val

class FrictionForce(Force):
    def __init__(self, value=vec3()):
        Force.__init__(self, value)
        self.friction = 0.9
    
    def get(self, object:Union['PhysicsComponent',None]=None):
        if object and object.vel.length_squared()>0:
            return -self.friction * object.vel.normalize()*(object.vel.length()**1.2) if object.get_world_position().z<1 else vec3()
        return vec3()

class GravityForce(Force):
    def __init__(self, strength:float=-2, axis=None):
        Force.__init__(self, (axis if axis else vec3(0, 0, 1))*strength)

class CollisionPoint:
    def __init__(self, a:None|vec3, b:None|vec3) -> None:
        self._a:vec3 = a if a else vec3()
        self._b:vec3 = b if b else vec3()
        self._normal:vec3=(self._b-self._a).normalize()
        self._depth:float=(self._b-self._a).length()
        pass

class Collision:
    def __init__(self, objectA:'PhysicsComponent', objectB:'PhysicsComponent', collision_point:CollisionPoint) -> None:
        self._objA = objectA
        self._objB = objectB
        self._collision_point = collision_point

class Solver:
    def __init__(self) -> None:
        pass

    def solve(self, collisions:list[Collision], dt:float):
        pass

class PhysicsComponent(DrawableComponent, SceneComponent):
    def __init__(self, parent, world : PhysicsWorld, pos=None, mass:float=1):
        SceneComponent.__init__(self, parent, pos)
        self.world : PhysicsWorld = world
        self.mass = mass
        self.vel = vec3()
        self.acc = vec3()
        self.simulate_physics = True

        self._bounding_box = BoundingBox(self._pos-self._size/2, self._pos+self._size/2)

        self.forces    :list[Force] = []
        self.one_forces:list[Force] = []

        world.register_physics_component(self)
    
    @SceneComponent.size.setter
    def size(self, s):
        self._size = s
        self._bounding_box = BoundingBox(self._pos-self.size/2, self._pos+self.size/2)
    
    def set_size(self, size:vec3):
        DrawableComponent.set_size(self, size)
        self._bounding_box = BoundingBox(self._pos-self.size/2, self._pos+self.size/2)
        return self

    def draw(self):
        Globals.game.draw_debug_rectangle(Globals.game.camera.world_to_screen(set_z(self._bounding_box._begin, 0)),
                                        Globals.game.camera.world_to_screen(set_z(self._bounding_box._end, 0)), color=vec3(255, 150, 0), thickness=1)
        pass
    
    def tick(self, dt:float):
        if not self.simulate_physics: return
        
        self.acc = vec3()
        for f in self.forces:
            force = f.get(self)
            self.acc += force
            Globals.game.draw_debug_vector(self._pos, self._pos+0.1*force)
        for f in self.one_forces:
            force = f.get(self)
            self.acc += force
            Globals.game.draw_debug_vector(self._pos, self._pos+0.1*force, (0,0,255))
        self.one_forces = []
        self.acc /= self.mass
        # log("Accélération : {}".format(self.acc))
        self.vel += self.acc * dt

        vel = self.vel * dt

        if vel.x<0:
            if self._pos.x+vel.x>self.world.limits[0].x:
                self._pos.x += vel.x
            else:
                self.vel.x = 0
                self._pos.x = self.world.limits[0].x
        
        if vel.x>0:
            if self._pos.x+vel.x<self.world.limits[1].x:
                self._pos.x += vel.x
            else: 
                self.vel.x = 0
                self._pos.x = self.world.limits[1].x
        
        if vel.y<0:
            if self._pos.y+vel.y>self.world.limits[0].y:
                self._pos.y += vel.y
            else:
                self.vel.y = 0
                self._pos.y = self.world.limits[0].y
        
        if vel.y>0:
            if self._pos.y+vel.y<self.world.limits[1].y:
                self._pos.y += vel.y
            else:
                self.vel.y = 0
                self._pos.y = self.world.limits[1].y

        if vel.z<0:
            if self._pos.z+vel.z>self.world.limits[0].z:
                self._pos.z += vel.z
            else:
                self.vel.z = 0
                self._pos.z = self.world.limits[0].z
        
        if vel.z>0:
            if self._pos.z+vel.z<self.world.limits[1].z:
                self._pos.z += vel.z
            else:
                self.vel.z = 0
                self._pos.z = self.world.limits[1].z
        
        Globals.game.draw_debug_vector(self._pos, self._pos+0.1*self.vel, (10,255,10))
        # Globals.game.draw_debug_box(self._pos-set_z(self.size/2, 0), self._pos+set_z(self.size/2, 0), (0, 0, 255), thickness=1)

class SpriteComponent(DrawableComponent):
    def __init__(self, parent, pos=vec3(), size=vec2(1, 1), image_name="default"):
        SceneComponent.__init__(self, parent=parent, pos=pos)
        Drawable.__init__(self)
        self.draw_size = size
        self.sprite = Globals.game.load_image(image_name, size=self.draw_size)
        self._size_locked = False
        self._draw_offset = vec2()
    
    def draw(self):
        Drawable.draw(self)
        if self.sprite:
            draw_pos = Globals.game.camera.world_to_screen(self.get_world_position())
            self.draw_size = Globals.game.camera.world_size2_to_screen(self.size.xy)
            Globals.game.draw_debug_box(self.get_world_position()-self.size/2, self.get_world_position()+self.size/2, (0, 255, 100))
            if self.sprite.size!=self.draw_size:
                if (not self._size_locked):
                    # log("Size if wrong, reloading sprite", logTypes.warning)
                    self.sprite = Globals.game.load_image(self.sprite.name, self.sprite.path, self.draw_size)
                else:
                    self.sprite.resize(self.draw_size)
            Globals.game.screen.blit(self.sprite.get_data(), draw_pos - self.draw_size.xy/2 + self._draw_offset)
    
    def set_draw_offset(self, offset:vec2):
        self._draw_offset = offset


class Actor(Object):
    def __init__(self, pos=vec3()):
        Object.__init__(self)
        self._root = SceneComponent(None, pos=pos)
    
    
    @property
    def root(self):
        return self._root

class Pawn(Actor):
    def __init__(self, world:PhysicsWorld, pos=vec3(), image_name="default"):
        Actor.__init__(self, pos=pos)
        self._root = PhysicsComponent(None, world, mass=0.1)
        self._root.forces = [GravityForce(-9.81), FrictionForce()]
        self.shadow = SpriteComponent(self.root, image_name="default_shadow")
        self.shadow.size = vec3(10, 10, 10)
        self.shadow.set_inherit_parent_location(False)
        self.shadow.set_draw_offset(vec2(0, 5))
        self.character = SpriteComponent(self.root, image_name=image_name)
    
    def update(self):        
        self.shadow._pos = vec3(self.root.get_world_position().x, self.root.get_world_position().y, Globals.world.line_trace(self.root.get_local_position(), vec3(0, 0, -1)).z)


def testSlimyEngine():
    pass

if __name__=="__main__":
    testSlimyEngine()