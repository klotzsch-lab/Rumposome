import matplotlib.pyplot as plt
import numpy as np

def gaussian(x, mu, sigma):
    """Returns the value of a Gaussian distribution with mean `mu` and standard deviation `sigma`."""
    return (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu)**2 / (2 * sigma**2))

def rotate_point(x, y, angle_degrees, origin=(0, 0)):
    """Rotates a point (x, y) around a given origin by angle (degrees)."""
    angle_radians = np.radians(angle_degrees)
    ox, oy = origin
    qx = ox + np.cos(angle_radians) * (x - ox) - np.sin(angle_radians) * (y - oy)
    qy = oy + np.sin(angle_radians) * (x - ox) + np.cos(angle_radians) * (y - oy)
    return qx, qy

def plot_hexagonal_grid_gaussian_distribution_fft(radius, rows, cols, accuracy, sigma):
    # Create a figure with five subplots (additional subplot for autocorrelation)
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(28, 6))

    # Calculate vertical and horizontal spacing between points
    vertical_spacing = np.sqrt(3) * radius
    horizontal_spacing = 1.5 * radius

    # List to store original (unrotated) coordinates
    original_x_coords = []
    original_y_coords = []

    # Plot the hexagonal grid on the first subplot (ax1)
    for row in range(rows):
        for col in range(cols):
            # Calculate the x and y position for each point
            x = col * horizontal_spacing
            y = row * vertical_spacing + (col % 2) * (vertical_spacing / 2)

            # Add random accuracy offset to the x and y positions
            x += np.random.uniform(-accuracy, accuracy)
            y += np.random.uniform(-accuracy, accuracy)

            # Store original (unrotated) coordinates
            original_x_coords.append(x)
            original_y_coords.append(y)

            # Plot each point in the hexagonal grid
            ax1.plot(x, y, 'o', markersize=8, color='blue')

    # Set equal scaling and remove axes for a clean appearance of the grid
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('Hexagonal Grid with Position Accuracy')

    # Plot the Gaussian intensity distribution for the unrotated grid
    x_values = np.linspace(min(original_x_coords) - 1, max(original_x_coords) + 1, 1000)
    intensity_profile = np.zeros_like(x_values)

    # Sum up Gaussian distributions for each unrotated x-coordinate
    for x in original_x_coords:
        intensity_profile += gaussian(x_values, x, sigma)

    # Plot the Gaussian intensity distribution on the second subplot (ax2)
    ax2.plot(x_values, intensity_profile, label="Gaussian Intensity", color="blue")
    ax2.set_title('Gaussian Intensity Distribution Along the x-axis')
    ax2.set_xlabel('x-axis')
    ax2.set_ylabel('Intensity')

    # Initialize an array to accumulate the Gaussian intensities over all rotations
    accumulated_intensity_profile = np.zeros_like(x_values)

    # Rotate the grid and accumulate the intensity profile for each rotation
    rotation_angles = np.arange(0, 360, 5)

    for angle in rotation_angles:
        rotated_x_coords = []

        # Rotate each point in the grid
        for x, y in zip(original_x_coords, original_y_coords):
            rx, _ = rotate_point(x, y, angle, origin=(0, 0))  # Only care about rotated x-coordinates
            rotated_x_coords.append(rx)

        # Compute the Gaussian intensity distribution for the rotated grid
        intensity_profile_rotated = np.zeros_like(x_values)
        for rx in rotated_x_coords:
            intensity_profile_rotated += gaussian(x_values, rx, sigma)

        # Accumulate the intensity profiles over all rotations
        accumulated_intensity_profile += intensity_profile_rotated

    # Plot the accumulated Gaussian intensity distribution on the third subplot (ax3)
    ax3.plot(x_values, accumulated_intensity_profile, label="Accumulated Intensity", color="blue")
    ax3.set_title('Accumulated Gaussian Intensity Over All Rotations')
    ax3.set_xlabel('x-axis')
    ax3.set_ylabel('Accumulated Intensity')

    # --- Compute the FFT of the accumulated intensity profile ---
    fft_values = np.fft.fft(accumulated_intensity_profile)
    fft_freqs = np.fft.fftfreq(len(accumulated_intensity_profile), d=(x_values[1] - x_values[0]))

    # Take the magnitude of the FFT and only plot the positive frequencies
    positive_freqs = fft_freqs[fft_freqs >= 0]
    fft_magnitude = np.abs(fft_values[:len(positive_freqs)])

    # Plot the FFT magnitude on the fourth subplot (ax4)
    ax4.plot(positive_freqs, fft_magnitude, label="FFT Magnitude", color="blue")
    ax4.set_title('FFT of Accumulated Intensity')
    ax4.set_xlabel('Frequency')
    ax4.set_ylabel('Magnitude')
    ax4.set_xlim([0.0, 0.1])
    ax4.set_ylim([0, 3000])

    # --- Compute the Autocorrelation of the accumulated intensity profile ---
    autocorrelation = np.correlate(accumulated_intensity_profile, accumulated_intensity_profile, mode='full')
    lags = np.arange(-len(accumulated_intensity_profile) + 1, len(accumulated_intensity_profile))

    # Plot the autocorrelation on the fifth subplot (ax5)
    ax5.plot(lags, autocorrelation, label="Autocorrelation", color="blue")
    ax5.set_title('Autocorrelation of Accumulated Intensity')
    ax5.set_xlabel('Lag')
    ax5.set_ylabel('Autocorrelation')
    ax5.set_xlim([0, len(accumulated_intensity_profile)])  # Focus on positive lags

    # Show all plots
    plt.tight_layout()
    plt.show()

# Define grid parameters: radius of dots, number of rows and columns
radius = 20
rows = 100
cols = 100
accuracy = 2  # Maximum offset for position accuracy
sigma = 10     # Standard deviation for Gaussian distributions

# Plot the hexagonal grid, Gaussian intensity distribution, accumulated intensity, FFT, and autocorrelation
plot_hexagonal_grid_gaussian_distribution_fft(radius, rows, cols, accuracy, sigma)
