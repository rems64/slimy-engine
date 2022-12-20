import math
import os
import random
import sys
import time
import colorit
from typing import List, Tuple, Union, Literal
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
    
class Colors:
    Color = pygame.Color
    black       = (0, 0, 0)
    white       = (255, 255, 255)
    green       = (0, 255, 0)
    red         = (255, 0, 0)
    darkgreen   = (150, 215, 140)

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
    
    def resize(self, size) -> None:
        assert self._data!=None
        self._data = pygame.transform.scale(self._data, size)
        self.size = size

    
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
    def __init__(self):
        if Globals.game: raise RuntimeError("There can exist only one game")
        Globals.game = self
        self.size = (640, 480)
        self.title = ""
        self._fonts : dict[str, pygame.font.Font] = {}
        self._clock : pygame.time.Clock = pygame.time.Clock()
        self._delta_time:float = 0.0
        self.__alive = True
        self.__target_fps = 60
        self.__background_color = Colors.black
        self.__events = []
        self.__images = {}
        self.active_scene : Scene = Scene()

        self.__frame_debugs = []
        self.debug_infos:dict[str, str] = {"fps": "0", "deltatime": "0"}
        self._no_debug = False

        self.load_image("default", "engine/default.png")
        self.load_image("default_shadow", "engine/default_shadow.png")
    
    def init(self, title="Slimy Engine"):
        pygame.init()
        flags = pygame.RESIZABLE | pygame.DOUBLEBUF
        self.screen = pygame.display.set_mode(self.size, flags)
        self.title = title
        pygame.display.set_caption(title)
        self._clock = pygame.time.Clock()
        self.load_font("debug_default", "engine/debug_font.ttf")
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
        if self.__images.get(name):
            if (not force_reload) and self.__images[name].get(get_image_size_tuple(size)):
                return self.__images[name][get_image_size_tuple(size)]
            else:
                if size:
                    p=self.resource_path(next(iter(self.__images[name].values())).path)
                    log("Loading image {} from disk with size ({}, {})".format(name, size[0], size[1]), logTypes.trace)
                    im = pygame.image.load(p).convert_alpha()
                    im = pygame.transform.scale(im, size)
                    s = im.get_size()
                    img = Image(name, vec2(s[0], s[1]), path, im)
                    self.__images[name][s] = img
                    return img
                else:
                    return max(self.__images[name])[1]
        if not path:
            raise RuntimeError("Never loaded this resource and no path specified ("+name+")")
        im = pygame.image.load(self.resource_path(path))
        log("Loading image {} from disk with {}".format(name, "size {}".format(size) if size else "default size"), logTypes.trace)
        if size:
            im = pygame.transform.scale(im, size)
        s = im.get_size()
        img = Image(name, vec2(s[0], s[1]), path, im)
        self.__images[name] = {}
        self.__images[name][s] = img
        return img

    def is_alive(self):
        return self.__alive
    
    def target_fps(self, fps):
        self.__target_fps = fps
        return self
    
    def get_delta_time(self) -> float:
        return self._delta_time
    
    def set_background_color(self, color):
        assert type(color)==tuple
        self.__background_color = color
        return self
    
    def on_resize(self, event):
        self.size = (event.dict['size'][0], event.dict['size'][1])
        self.camera.update_screen_size(event.dict['size'])
        pass
    
    def begin_frame(self, dont_clear=False):
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                self.__alive = False
            if event.type == pygame.VIDEORESIZE:
                self.on_resize(event)
                pygame.display.update()
        if not dont_clear:
            self.screen.fill(self.__background_color)
    
    def end_frame(self):
        self._delta_time = self._clock.get_time()
        if not self._no_debug:
            for debug in self.__frame_debugs:
                debug.draw(self.screen)
            self.__frame_debugs = []
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
        self._clock.tick(self.__target_fps)
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
            self.__frame_debugs.append(vector)
        
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
            self.__frame_debugs.append(square)

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
            self.__frame_debugs.append(square)
    
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


class Scene:
    def __init__(self):
        self._objects : List[SceneComponent] = []
        self.manual_rendering : bool = False
        self.active_camera : Camera = OrthographicCamera() # type: ignore
        self._drawables : SortedList = SortedList()
        self._tilemaps : list[Tilemap] = []
        self._tilesets : list[Tileset] = []
        self._backgrounds : list[SpriteComponent] = []
    
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
    
    def draw(self):
        if not self.manual_rendering:
            for background in self._backgrounds:
                background.draw()
            for obj in self._drawables:
                obj.draw()
    
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

    def tick(self):
        self.tmp_tick=time.time_ns()
        dt = (self.tmp_tick-self.last_tick)*1.0E-9
        self.last_tick=self.tmp_tick
        if self._draw_borders and self.limits[0].length_squared()<math.inf and self.limits[1].length_squared()<math.inf:
            Globals.game.draw_debug_box(set_z(self.limits[0], 0), set_z(self.limits[1], 0), vec3(255, 0, 0), thickness=2)
        
        for obj in self.objects:
            obj.tick(dt)
    
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
        if object and object.vel.length()>0:
            return -self.friction * object.vel.normalize()*(object.vel.length()**1.2) if object.get_world_position().z<1 else vec3()
        return vec3()

class GravityForce(Force):
    def __init__(self, strength:float=-2):
        super().__init__(vec3(0, 0, strength))

class PhysicsComponent(DrawableComponent, SceneComponent):
    def __init__(self, parent, world : PhysicsWorld, pos=None, mass:float=1):
        SceneComponent.__init__(self, parent, pos)
        self.world : PhysicsWorld = world
        self.mass = mass
        self.vel = vec3()
        self.acc = vec3()
        self.simulate_physics = True

        self.forces    :list[Force] = []
        self.one_forces:list[Force] = []

        world.register_physics_component(self)
    
    def draw(self):
        Globals.game.draw_debug_rectangle(Globals.game.camera.world_to_screen(self._pos-vec3(5, 5, 0)),
                                        Globals.game.camera.world_to_screen(self._pos+vec3(5, 5, 0)), color=vec3(255, 150, 0), thickness=1)
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
        Globals.game.draw_debug_box(self._pos-set_z(self.size/2, 0), self._pos+set_z(self.size/2, 0), (0, 0, 255), thickness=1)

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


class Solver:
    def __init__(self) -> None:
        pass

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