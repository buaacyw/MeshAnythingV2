from PIL import Image

# Input and output file names
input_gif = 'demo/demo_video.gif'
output_gif = 'demo_video.gif'

# Open the GIF file
with Image.open(input_gif) as gif:
    # Get the original width and height
    original_width, original_height = gif.size
    # Calculate the scale to resize the width to 512
    scale = 512 / original_width
    new_width = 512
    new_height = int(original_height * scale)

    # List to store the resized frames
    frames = []
    try:
        while True:
            # Resize each frame
            resized_frame = gif.copy().resize((new_width, new_height), Image.LANCZOS)
            frames.append(resized_frame)
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass  # End of frames

    # Save the resized frames as a new GIF
    frames[0].save(output_gif, save_all=True, append_images=frames[1:], loop=0, duration=gif.info['duration'], disposal=2)
