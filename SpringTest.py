from engine.slimyengine import *

class Ball(PhysicsComponent, DrawableComponent):
    def __init__(self, parent, world: PhysicsWorld, pos=None, mass: float = 1):
        PhysicsComponent.__init__(self, parent, world, pos, mass)
        DrawableComponent.__init__(self, parent, pos)

game = Game((1280, 960)).init().target_fps(60).set_background_color(Colors.darkgreen).set_debug(True)

scene = Scene().set_ambient_light(vec3(1, 1, 1))
game.load_scene(scene)
camera = scene.active_camera

world = PhysicsWorld()

world.set_limits(vec3(-math.inf, -math.inf, 0.), vec3(math.inf, math.inf, 20.))
Globals.world = world

game.load_image("player", "player.png")
game.load_image("bush", "ball.png")

ball1 = PhysicsComponent(None, world, vec3(0,  5, 0), 1).set_size(vec3(1, 1, 1))
ball2 = PhysicsComponent(None, world, vec3(0, -5, 0), 1)
scene.register_component(ball1).register_component(ball2)

ball1.forces = [GravityForce(9.81, vec3(0, 1, 0))]


while game.is_alive():
    game.begin_frame()
    game.draw_debug_spring(ball1.get_world_position(), ball2.get_world_position())
    world.tick()
    scene.update()
    scene.draw()
    scene.light_pass()
    game.end_frame()