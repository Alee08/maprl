Nuovo
+95
-0

"""Shared helpers for environment setup across concurrent scenarios."""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from unified_planning.shortcuts import Object, UserType


Coordinate = Tuple[int, int]
Room = Iterable[Coordinate]


def build_object_positions(
    coordinates: Dict[str, List[Coordinate]],
    walls: Sequence[Tuple[Coordinate, Coordinate]],
    extra: Optional[Dict[str, List[Coordinate]]] = None,
) -> Dict[str, List[Coordinate]]:
    """Create base object positions and merge optional extras."""

    base_positions = {
        "plant": coordinates["plant"],
        "coffee": coordinates["coffee"],
        "letter": coordinates["letter"],
        "office_walls": walls,
    }

    if extra:
        base_positions.update(extra)

    return base_positions


def find_connector_between_rooms(
    room_a: Room, room_b: Room, walls_set: Iterable[Tuple[Coordinate, Coordinate]]
) -> Optional[Tuple[Coordinate, Coordinate]]:
    """Return the first pair of adjacent cells between two rooms without a wall."""

    room_b_cells = set(room_b)
    for x, y in room_a:
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            neighbor = (x + dx, y + dy)
            if neighbor in room_b_cells and ((x, y), neighbor) not in walls_set:
                return (x, y), neighbor
    return None


def build_connectors(
    pairs: Iterable[Tuple[str, str]],
    rooms: Dict[str, Room],
    walls: Sequence[Tuple[Coordinate, Coordinate]],
) -> List[Tuple[Coordinate, Coordinate]]:
    """Build drawable connectors for the provided room pairs."""

    walls_set = set()
    for cell_a, cell_b in walls:
        walls_set.add((cell_a, cell_b))
        walls_set.add((cell_b, cell_a))

    connectors = []
    for room_a, room_b in pairs:
        if room_a not in rooms or room_b not in rooms:
            continue

        connector_cells = find_connector_between_rooms(
            rooms[room_a], rooms[room_b], walls_set
        )
        if connector_cells:
            connectors.append(connector_cells)

    return connectors


def generate_grid_locations_and_coordinates(grid_size: int):
    """Generates locations and corresponding coordinates for a grid of given size."""

    Location = UserType("Location")

    locations = []  # List to track created locations
    coordinates = []  # List to track coordinates and corresponding locations

    # Loop through grid rows and columns to generate locations and coordinates
    for row in range(1, grid_size + 1):
        for col in range(1, grid_size + 1):
            # Create location name based on row and column
            location_name = f"l{row}{col}"

            # Create a Location object with the generated name
            location = Object(location_name, Location)

            # Add the Location object to the list of locations
            locations.append(location)

            # Add the coordinate-location pair to the list of coordinates
            coordinates.append(((row - 1, col - 1), location))

    return locations, coordinates
