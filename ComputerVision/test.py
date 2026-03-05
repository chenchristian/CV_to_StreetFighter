import pygame

pygame.init()

# Create a small window to capture keyboard input
screen = pygame.display.set_mode((200, 200))
pygame.display.set_caption("Key Press Debug")

running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Get current key states
    keys = pygame.key.get_pressed()
    
    # Print the length of the tuple
    print(f"Length of key.get_pressed(): {len(keys)}")
    
    # Optional: print only pressed keys
    pressed_keys = [i for i, val in enumerate(keys) if val]
    print(f"Pressed key indices: {pressed_keys}")

    # Limit to ~10 frames per second
    clock.tick(10)

pygame.quit()