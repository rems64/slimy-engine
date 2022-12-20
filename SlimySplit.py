from engine.slimyengine import *

class Player(Pawn):
    def __init__(self, world:PhysicsWorld):
        Pawn.__init__(self, world, image_name="player")
    
    def jump(self):
        if self._root.get_world_position().z<1:
            self._root.one_forces.append(Force(vec3(0, 0, 100)))

    def update(self):
        Pawn.update(self)
        keys = pygame.key.get_pressed()
        force = Force(vec3())
        if keys[pygame.K_LEFT]: # We can check if a key is pressed like this
            force.x += -1
        
        if keys[pygame.K_RIGHT]:
            force.x += 1

        if keys[pygame.K_UP]:
            force.y += -1

        if keys[pygame.K_DOWN]:
            force.y += 1
        
        if keys[pygame.K_SPACE]:
            self.jump()
        
        if keys[pygame.K_g]:
            l = 0.5 if Globals.game.camera.get_zoom()==1 else 1
            Globals.game.camera.set_zoom(l)
            Globals.game.debug_infos["zoom"]=str(l)

        if force.length>0:
            fac=10.
            if self._root.get_world_position().z>1:
                # Air control
                fac=1.
            force.normalize_ip()
            force.scale(fac)
            self._root.one_forces.append(force)

class Bush(Actor):
    def __init__(self, pos):
        Actor.__init__(self, pos=pos)
        self.bush = SpriteComponent(self.root, image_name="bush")
        self.bush.size = vec2(3, 3)


# Creating global game variable (registers itself in Globals static class)
game = Game().init().target_fps(120).set_background_color(Colors.darkgreen).set_debug(True)

# Main scene, load it and create an alias for later access
scene = Scene()
game.load_scene(scene)
camera = scene.active_camera

# Define the PhysicsWorld, which manages everything related to physics
world = PhysicsWorld()
# world.set_limits(vec3(-10., -10., 0.), vec3(10., 10., 20.))
world.set_limits(vec3(-math.inf, -math.inf, 0.), vec3(math.inf, math.inf, 20.))
Globals.world = world

# Resources loading, each resource has to be given a unique name (used by the cache)
game.load_image("player", "player.png")
game.load_image("bush", "ball.png")
scene.load_map("level2", "level2.json")

# Finally create the player, defined earlier
player = Player(world=world)
# Maybe the object could register itself on creation, but the choice is left to the user for now
scene.register_component(player.root)
# Deprecated....but no alternative yet :))
player.root.set_local_position(vec3(0., 0., 2.5))
player.root.size = vec3(2.0, 2.0, 2.)
player.character.size=player.root.size
player.shadow.size=player.root.size

camera.set_local_position(vec3(0, 0, 0))

# Main gameloop
while game.is_alive():
    # Must be called before everything else
    game.begin_frame()
    # Update physics => apply forces from the last frame, this needs a rework in order to react instantly to user inputs
    world.tick()
    # Update player input
    player.update()
    # Update parent/ child transforms, quite effective for now but may require a rewrite later
    scene.update()
    # Makes the camera follow the player
    camera._pos = set_z(player.root.get_world_position(), 0) # Math.lerp_squared(camera.pos, player.pos.xy, 0.001)
    # Finally draw the SpriteComponent (every objects which inherits from Drawable and is registered in scene)
    scene.draw()
    # Performs all debug draws, update delta time and wait idle until frame time is reached
    game.end_frame()