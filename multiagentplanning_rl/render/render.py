import pygame
import cv2
import os
import imageio


class EnvironmentRenderer:
    def __init__(
        self,
        grid_width,
        grid_height,
        agents,
        object_positions,
        goals,
        in_cell_size,
        cell_size=1,
        resource_overrides=None,
    ):
        self.grid_width = grid_width  # Number of rooms horizontally
        self.grid_height = grid_height  # Number of rooms vertically
        self.agents = agents
        self.object_positions = object_positions or {}  # Positions of objects
        self.goals = goals
        self.in_cell_size = in_cell_size  # Number of cells inside a room (e.g., 3 for a 3x3 grid inside each room)
        self.cell_size = cell_size  # Size of a room in terms of cells
        self.agent_images = {}  # Cache for loaded agent images
        self.resources = {}  # Dictionary for loaded resources
        self.frames = []  # List to store frames for video rendering
        self.resource_overrides = resource_overrides or {}

        # Calculate the image path
        self.img_path = os.path.dirname(__file__)
        self.init_pygame()

    def load_resource(self, path, size):
        """Load and resize an image from memory."""
        full_path = os.path.join(self.img_path, path)
        image = pygame.image.load(full_path).convert_alpha()
        return pygame.transform.scale(image, size)

    def init_pygame(self):
        pygame.init()
        self.inner_cell_size = 100  # Pixel size of each inner cell (adjust as needed)
        self.frames = []
        self.font = pygame.font.SysFont("Arial", 25)

        # Calculate total grid size in pixels
        self.screen_width = self.grid_width * self.in_cell_size * self.inner_cell_size
        self.screen_height = self.grid_height * self.in_cell_size * self.inner_cell_size
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        # Dictionary of resources with paths (adjust sizes as needed)
        resource_info = {
            "colosseo": ("img/colosseo.png", (90, 90)),
            "piazza": ("img/piazza.png", (90, 90)),
            "bcn": ("img/bcn.png", (95, 95)),
            "madrid": ("img/mdn.png", (90, 90)),
            "battlo": ("img/battlo.png", (90, 90)),
            "piazza_di_spagna": ("img/piazza_di_spagna2.png", (95, 95)),
            "ita_man": ("img/ita_man.png", (85, 85)),
            "bcn_man": ("img/bcn_man2.png", (80, 80)),
            "CR7": ("img/CR7.png", (70, 70)),
            "juve": ("img/juve.png", (75, 75)),
            "holes": ("img/hole.png", (90, 90)),
            "ponte_immagine": ("img/ponte_.png", (40, 40)),
            "barca_a_remi": ("img/barca_.png", (40, 40)),
            "plant": (
                "img/pianta.png",
                (self.inner_cell_size, self.inner_cell_size),
            ),  # office_world
            # "plant": ("img/albero.png", (self.inner_cell_size, self.inner_cell_size)),
            # "plant": ("img/buco_lab.png", (self.inner_cell_size, self.inner_cell_size)),
            # "coffee": ("img/key_2.png", (self.inner_cell_size-6, self.inner_cell_size-6)), #tempio
            # "letter": ("img/tesoro.png", (self.inner_cell_size-3, self.inner_cell_size-3)), #tempio
            "coffee": (
                "img/coffee.png",
                (self.inner_cell_size - 6, self.inner_cell_size - 6),
            ),  # office_world
            "letter": (
                "img/email.png",
                (self.inner_cell_size - 3, self.inner_cell_size - 3),
            ),  # office_world
            # "coffee": ("img/torcia.png", (self.inner_cell_size - 6, self.inner_cell_size - 6), ),  # maze
            # "letter": ("img/remi.png", (self.inner_cell_size, self.inner_cell_size),),  # maze
            # "remi": ("img/remi.png", (self.inner_cell_size, self.inner_cell_size)),
        }

        for name, resource in self.resource_overrides.items():
            if callable(resource):
                resource_info[name] = resource(self)
            else:
                resource_info[name] = resource

        # Load all resources defined in resource_info
        for name, (path, size) in resource_info.items():
            self.resources[name] = self.load_resource(path, size)

    def load_agent_image(self, agent_type):
        # Dictionary mapping agent types to image paths and sizes
        # office/temple
        """image_map = {
            "a1": ("img/ita_man.png", (self.inner_cell_size, self.inner_cell_size)),
            "a2": ("img/juve.png", (self.inner_cell_size, self.inner_cell_size)),
            "a3": ("img/bcn_man2.png", (self.inner_cell_size, self.inner_cell_size)), #office
            "a4": ("img/CR7.png", (self.inner_cell_size, self.inner_cell_size)),      #office
            "a5": ("img/juve.png", (self.inner_cell_size, self.inner_cell_size)),
            "a6": ("img/we.png", (self.inner_cell_size-5, self.inner_cell_size-5)),
            "a7": ("img/o.png", (self.inner_cell_size-5, self.inner_cell_size-5)),
            "a8": ("img/o.png", (self.inner_cell_size-5, self.inner_cell_size-5)),
            "a9": ("img/o.png", (self.inner_cell_size-5, self.inner_cell_size-5)),
            "a10": ("img/we.png", (self.inner_cell_size-5, self.inner_cell_size-5)),
            # Add more mappings if needed...
        }"""
        # maze
        image_map = {
            "a1": ("img/o.png", (self.inner_cell_size - 5, self.inner_cell_size - 5)),
            "a2": ("img/juve.png", (self.inner_cell_size, self.inner_cell_size)),
            "a3": ("img/o.png", (self.inner_cell_size - 5, self.inner_cell_size - 5)),
            "a4": ("img/o.png", (self.inner_cell_size - 5, self.inner_cell_size - 5)),
            "a5": ("img/juve.png", (self.inner_cell_size, self.inner_cell_size)),
            "a6": ("img/ita_man.png", (self.inner_cell_size, self.inner_cell_size)),
            "a7": ("img/ita_man.png", (self.inner_cell_size, self.inner_cell_size)),
            "a8": ("img/ita_man.png", (self.inner_cell_size, self.inner_cell_size)),
            "a9": ("img/o.png", (self.inner_cell_size - 5, self.inner_cell_size - 5)),
            "a10": ("img/we.png", (self.inner_cell_size - 5, self.inner_cell_size - 5)),
            # Add more mappings if needed...
        }

        # Default image path and size
        default_image_path = "img/bcn_man2.png"
        default_image_size = (self.inner_cell_size, self.inner_cell_size)

        # Check if the agent image has already been loaded
        if agent_type not in self.agent_images:
            try:
                # Load the specific agent image or the default if not present
                if agent_type in image_map:
                    file_name, size = image_map[agent_type]
                else:
                    # Use default image
                    file_name, size = default_image_path, default_image_size

                # Construct the full image path
                full_path = os.path.join(self.img_path, file_name)
                image = pygame.image.load(full_path).convert_alpha()
                self.agent_images[agent_type] = pygame.transform.scale(image, size)

            except pygame.error as e:
                print(f"Error loading image for '{agent_type}': {e}")
                self.agent_images[agent_type] = None  # Or another default image

        return self.agent_images[agent_type]

    def get_agent_image(self, agent_name, small=False):
        # Use `load_agent_image` to get the image based on the agent type
        agent_type = agent_name  # Assuming `agent_name` corresponds to the agent type for simplicity
        image = self.load_agent_image(agent_type)
        if image and small:
            # Resize the image for multiple agents in the same cell
            small_size = (self.inner_cell_size // 2, self.inner_cell_size // 2)
            image = pygame.transform.scale(image, small_size)
        return image

    def render(self, episode, obs):
        self.screen.fill((255, 255, 255))

        total_grid_width = self.grid_width * self.in_cell_size
        total_grid_height = self.grid_height * self.in_cell_size

        # Draw the grid lines for the entire grid (including inner cells)
        for x in range(total_grid_width + 1):
            pygame.draw.line(
                self.screen,
                (200, 200, 200),
                (x * self.inner_cell_size, 0),
                (x * self.inner_cell_size, self.screen_height),
            )
        for y in range(total_grid_height + 1):
            pygame.draw.line(
                self.screen,
                (200, 200, 200),
                (0, y * self.inner_cell_size),
                (self.screen_width, y * self.inner_cell_size),
            )

        for goal_name, (x, y) in self.goals.items():
            # Calculate pixel position
            pos_x = x * self.inner_cell_size
            pos_y = y * self.inner_cell_size

            # Create a rectangle for the goal cell
            goal_rect = pygame.Rect(
                pos_x, pos_y, self.inner_cell_size, self.inner_cell_size
            )

            # Fill the cell with a distinctive color (e.g., light green)
            pygame.draw.rect(self.screen, (255, 215, 0), goal_rect)

            # Optionally, add the goal name
            text_surface = self.font.render(goal_name, True, (0, 0, 0))
            text_rect = text_surface.get_rect(center=goal_rect.center)
            self.screen.blit(text_surface, text_rect)

        # Draw the walls
        for (cell1, cell2) in self.object_positions.get("office_walls", []):
            x1, y1 = cell1
            x2, y2 = cell2

            if x1 == x2 and abs(y1 - y2) == 1:
                # Horizontal wall
                x_start = x1 * self.inner_cell_size
                x_end = (x1 + 1) * self.inner_cell_size
                y = (
                    max(y1, y2) * self.inner_cell_size
                )  # Wall at the top of the lower cell
                pygame.draw.line(self.screen, (0, 0, 0), (x_start, y), (x_end, y), 5)
            elif y1 == y2 and abs(x1 - x2) == 1:
                # Vertical wall
                y_start = y1 * self.inner_cell_size
                y_end = (y1 + 1) * self.inner_cell_size
                x = (
                    max(x1, x2) * self.inner_cell_size
                )  # Wall on the left of the right cell
                pygame.draw.line(self.screen, (0, 0, 0), (x, y_start), (x, y_end), 5)
            else:
                print(f"Invalid wall between {cell1} and {cell2}")

        # Draw the objects using absolute coordinates
        for obj_type in ["plant", "coffee", "letter"]:
            for x, y in self.object_positions.get(obj_type, []):
                pos_x = x * self.inner_cell_size
                pos_y = y * self.inner_cell_size

                obj_image = self.resources.get(obj_type)
                if obj_image:
                    self.screen.blit(obj_image, (pos_x, pos_y))

        # Collect agent positions
        agent_positions = {}

        # Gather position information for all agents
        for agent_name, agent_state in obs.items():
            x = agent_state.get((agent_name, "pos_x"))
            y = agent_state.get((agent_name, "pos_y"))
            i = agent_state.get((agent_name, "pos_i"))
            j = agent_state.get((agent_name, "pos_j"))

            position = (x, y, i, j)
            if position not in agent_positions:
                agent_positions[position] = []
            agent_positions[position].append(agent_name)

        # Draw the agents
        for position, agents_at_pos in agent_positions.items():
            x, y, i, j = position
            # Calculate absolute position in the grid
            abs_x = x * self.in_cell_size + i
            abs_y = y * self.in_cell_size + j

            pos_x = abs_x * self.inner_cell_size
            pos_y = abs_y * self.inner_cell_size

            if len(agents_at_pos) > 1:
                # Multiple agents at the same position
                num_agents = len(agents_at_pos)
                small_size = (self.inner_cell_size // 2, self.inner_cell_size // 2)
                for index, agent_name in enumerate(agents_at_pos):
                    agent_image = self.get_agent_image(agent_name, small=True)
                    if agent_image:
                        # Calculate offset to arrange images within the cell
                        offset_x = (index % 2) * small_size[0]
                        offset_y = (index // 2) * small_size[1]
                        self.screen.blit(
                            agent_image, (pos_x + offset_x, pos_y + offset_y)
                        )
            else:
                # Only one agent at this position
                agent_name = agents_at_pos[0]
                agent_image = self.get_agent_image(agent_name)
                if agent_image:
                    self.screen.blit(agent_image, (pos_x, pos_y))

        pygame.display.flip()

        # Capture the frame for video rendering
        frame = pygame.surfarray.array3d(self.screen).transpose([1, 0, 2])
        self.frames.append(frame)

    def save_episode(self, episode):
        # if episode == 0: #Save first frame
        # self.save_first_frame("maze.png")
        # Create the "episodes" folder if it doesn't exist
        episodes_dir = "episodes"
        os.makedirs(episodes_dir, exist_ok=True)

        if self.frames:
            # Save as AVI video
            video_path = f"episodes/episode_{episode}.avi"
            height, width, layers = self.frames[0].shape
            video = cv2.VideoWriter(
                video_path, cv2.VideoWriter_fourcc(*"DIVX"), 2, (width, height)
            )

            for frame in self.frames:
                video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            video.release()

            # Save as GIF
            gif_path = f"episodes/episode_{episode}.gif"
            imageio.mimsave(gif_path, self.frames, fps=2, loop=0)

            self.frames = []  # Clear the frames list

    def save_first_frame(self, filename="first_frame.png"):
        """Save the first rendered frame as an image."""
        if self.frames:
            first_frame = self.frames[0]
            # Convert the frame to a Pygame surface
            surface = pygame.surfarray.make_surface(first_frame.transpose([1, 0, 2]))
            # Save the surface as a PNG image
            pygame.image.save(surface, filename)
            print(f"First frame saved as {filename}")
        else:
            print("No frame available to save.")

    def simulate_agents(self, agent_paths):
        max_steps = max(len(path) for path in agent_paths.values())
        for step in range(max_steps):
            obs = {}
            for agent, path in agent_paths.items():
                if step < len(path):
                    x, y, i, j = path[step]
                    obs[agent] = {
                        (agent, "pos_x"): x,
                        (agent, "pos_y"): y,
                        (agent, "pos_i"): i,
                        (agent, "pos_j"): j,
                    }
                else:
                    x, y, i, j = path[-1]
                    obs[agent] = {
                        (agent, "pos_x"): x,
                        (agent, "pos_y"): y,
                        (agent, "pos_i"): i,
                        (agent, "pos_j"): j,
                    }
            self.render(step, obs)
        self.save_episode("final_simulation")
