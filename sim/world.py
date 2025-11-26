# ...existing code...
import pybullet as p

def load_world():
    p.setGravity(0, 0, -9.81)
    plane = p.loadURDF("plane.urdf")

    def create_box(center, half_extents, color=[0.6, 0.3, 0.2, 1.0]):
        """Cria um corpo estático em forma de caixa (colisão + visual)."""
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
        p.createMultiBody(baseCollisionShapeIndex=collision,
                          baseVisualShapeIndex=visual,
                          basePosition=center)

    def create_wall_with_door(center, length, thickness, height,
                              door_center: float = None, door_width: float = 1.0,
                              horizontal: bool = True, color=[0.6, 0.3, 0.2, 1.0]):
        """
        Cria uma parede (caixa) possivelmente dividida por uma porta.
        - center: [x,y,z] centro da parede
        - length: comprimento total da parede (no eixo da parede)
        - thickness: espessura (perpendicular ao comprimento)
        - height: altura da parede
        - door_center: posição da porta em relação ao centro da parede (m). None => sem porta
        - door_width: largura da porta (m)
        - horizontal: True se a parede estende-se no eixo X, False => eixo Y
        """
        cx, cy, cz = center
        if door_center is None:
            # parede inteira
            if horizontal:
                half_extents = [length / 2.0, thickness / 2.0, height / 2.0]
                create_box([cx, cy, cz], half_extents, color)
            else:
                half_extents = [thickness / 2.0, length / 2.0, height / 2.0]
                create_box([cx, cy, cz], half_extents, color)
            return

        # com porta: calculamos segmentos à esquerda e direita (ou inferior/superior)
        L = length
        dw = door_width
        dc = door_center  # medido a partir do centro da parede

        # comprimento à esquerda da porta (entre a borda esquerda e a borda da porta)
        left_len = (L / 2.0) + dc - (dw / 2.0)
        right_len = L - (left_len + dw)

        # left segment
        if left_len > 0.001:
            if horizontal:
                hx = left_len / 2.0
                hy = thickness / 2.0
                hz = height / 2.0
                left_cx = cx - (L / 2.0) + hx
                create_box([left_cx, cy, cz], [hx, hy, hz], color)
            else:
                hx = thickness / 2.0
                hy = left_len / 2.0
                hz = height / 2.0
                left_cy = cy - (L / 2.0) + hy
                create_box([cx, left_cy, cz], [hx, hy, hz], color)

        # right segment
        if right_len > 0.001:
            if horizontal:
                hx = right_len / 2.0
                hy = thickness / 2.0
                hz = height / 2.0
                right_cx = cx + (L / 2.0) - hx
                create_box([right_cx, cy, cz], [hx, hy, hz], color)
            else:
                hx = thickness / 2.0
                hy = right_len / 2.0
                hz = height / 2.0
                right_cy = cy + (L / 2.0) - hy
                create_box([cx, right_cy, cz], [hx, hy, hz], color)

    def create_room(center_xy, size_x, size_y, wall_thickness=0.1, wall_height=2.0,
                    doors=None):
        """
        Cria as quatro paredes de um cômodo retangular.
        - center_xy: (x,y) do centro do cômodo no chão
        - size_x, size_y: dimensões do cômodo (largura X, comprimento Y)
        - doors: dict com chaves 'north','south','east','west' contendo
                 door_center_relative (m) ou None. door_center_relative é
                 deslocamento da porta em relação ao centro da parede.
        """
        cx, cy = center_xy
        z = wall_height / 2.0  # centro Z das paredes

        if doors is None:
            doors = {'north': None, 'south': None, 'east': None, 'west': None}

        # north wall (y positive side)
        north_center = [cx, cy + size_y / 2.0, z]
        create_wall_with_door(north_center, length=size_x, thickness=wall_thickness, height=wall_height,
                              door_center=doors.get('north'), door_width=1.0, horizontal=True)

        # south wall (y negative side)
        south_center = [cx, cy - size_y / 2.0, z]
        create_wall_with_door(south_center, length=size_x, thickness=wall_thickness, height=wall_height,
                              door_center=doors.get('south'), door_width=1.0, horizontal=True)

        # east wall (x positive side) - vertical
        east_center = [cx + size_x / 2.0, cy, z]
        create_wall_with_door(east_center, length=size_y, thickness=wall_thickness, height=wall_height,
                              door_center=doors.get('east'), door_width=1.0, horizontal=False)

        # west wall (x negative side) - vertical
        west_center = [cx - size_x / 2.0, cy, z]
        create_wall_with_door(west_center, length=size_y, thickness=wall_thickness, height=wall_height,
                              door_center=doors.get('west'), door_width=1.0, horizontal=False)

    # --- Layout: sala central conectada a quartos/cozinha ---
    # sala central 5x5 com portas norte/sul (alinhadas)
    create_room(center_xy=(0.0, 0.0), size_x=5.0, size_y=5.0, doors={
        'north': 0.0,  # porta central na parede norte
        'south': 0.0,  # porta central na parede sul
        'east': 0.0,
        'west': 0.0
    })

    # quarto norte 3x3, porta para sala (na parede sul do quarto)
    # pos do quarto: deslocado para cima (centro y = -4.0 no arquivo antigo era -4); aqui usamos y = -4.0 para mesma orientação
    create_room(center_xy=(0.0, -4.0), size_x=3.0, size_y=3.0, doors={
        'north': 0.0,
        'south': None,  # porta central conectando à sala
        'east': None,
        'west': None
    })

    # cozinha sul 3x3, porta para sala
    create_room(center_xy=(0.0, 4.0), size_x=3.0, size_y=3.0, doors={
        'north': None,  # porta central conectando à sala
        'south': 0.0,
        'east': None,
        'west': None
    })

    # opcional: pequenos armários laterais, conectados à sala à esquerda e direita
    create_room(center_xy=(-4.0, 0.0), size_x=1.6, size_y=2.0, doors={
        'east': 0.0,  # porta ligando à sala (west wall da sala)
        'north': None,
        'south': None,
        'west': None
    })
    create_room(center_xy=(4.0, 0.0), size_x=1.6, size_y=2.0, doors={
        'west': 0.0,  # porta ligando à sala (east wall da sala)
        'north': None,
        'south': None,
        'east': None
    })

    return plane
# ...existing code...