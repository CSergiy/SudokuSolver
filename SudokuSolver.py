import io

import cv2
import numpy as np
import pyautogui
from PIL import Image
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
import tensorflow as tf

import streamlit as st




def preprocess_image(image):
    """
    Apply preprocessing steps to invert the image colors and make it similar to MNIST format.
    
    Parameters:
        image (PIL.Image.Image): The input image to preprocess.
    
    Returns:
        preprocessed_image (numpy.ndarray): The preprocessed image ready for further analysis.
    """
    # Convert the PIL Image to an OpenCV image (PIL uses RGB, OpenCV uses BGR)
    open_cv_image = np.array(image)
    # Convert RGB to BGR 
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to smooth out the image
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Apply adaptive thresholding to create a binary image
    # This step inverts the image: background becomes black, digits become white
    threshold_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY_INV, 11, 2)
    
    # No need for edge detection, as we want to keep the digits filled
    
    return threshold_image

def detect_sudoku_grid(preprocessed_image):
    """
    Detect the Sudoku grid and divide it into 9x9 cells.
    
    Parameters:
        preprocessed_image (numpy.ndarray): The preprocessed image of the Sudoku puzzle.
        
    Returns:
        cells (list of numpy.ndarray): List of 9x9 cell images of the Sudoku grid.
    """
    # Step 1: Find contours
    contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Assume the largest contour is the Sudoku grid
    max_area = 0
    sudoku_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            sudoku_contour = contour
    
    if sudoku_contour is None:
        print("Sudoku grid not found")
        return None
    
    # Step 2: Apply perspective transform to get a top-down view of the Sudoku grid
    # Find the corners of the grid
    peri = cv2.arcLength(sudoku_contour, True)
    approx = cv2.approxPolyDP(sudoku_contour, 0.02 * peri, True)
    
    if len(approx) != 4:
        print("Could not find corners of the grid")
        return None
    
    # Order the corners
    ordered_corners = order_points(np.squeeze(approx))
    
    # The maximum width and height of the grid
    side = max([
        np.linalg.norm(ordered_corners[0] - ordered_corners[1]),
        np.linalg.norm(ordered_corners[1] - ordered_corners[2]),
        np.linalg.norm(ordered_corners[2] - ordered_corners[3]),
        np.linalg.norm(ordered_corners[3] - ordered_corners[0])
    ])
    
    # Destination points for the perspective transform
    dst = np.array([
        [0, 0],
        [side - 1, 0],
        [side - 1, side - 1],
        [0, side - 1]], dtype="float32")
    
    # Apply perspective transform
    M = cv2.getPerspectiveTransform(ordered_corners, dst)
    warped = cv2.warpPerspective(preprocessed_image, M, (int(side), int(side)))
    
    # Step 3: Divide the warped image into 9x9 cells
    cells = []
    cell_size = side / 9
    for i in range(9):
        row = []
        for j in range(9):
            start_x = int(j * cell_size)
            start_y = int(i * cell_size)
            cell = warped[start_y:start_y + int(cell_size), start_x:start_x + int(cell_size)]
            row.append(cell)
        cells.append(row)
    
    return cells

def order_points(pts):
    """
    Order the points in clockwise order starting from the top-left point.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def crop_to_central_area(cell, border_size=4):
    """
    Crops the cell image to remove a border of `border_size` pixels from all sides.
    
    Parameters:
    - cell: A 28x28 numpy array representing the cell image.
    - border_size: The width of the border to remove from each side.
    
    Returns:
    - A cropped cell image.
    """
    cropped_cell = cell[border_size:-border_size, border_size:-border_size]
    return cropped_cell

def crop_cells_grid(cells, border_size=6):
    """
    Crop the borders from each cell in a 9x9 grid.
    
    Parameters:
    - cells: A 9x9 grid of cell images.
    - border_size: The width of the border to remove from each side.
    
    Returns:
    - A 9x9 grid of cropped cell images.
    """
    cropped_cells = []
    for row in cells:
        cropped_row = [crop_to_central_area(cell, border_size) for cell in row]
        cropped_cells.append(cropped_row)
    return cropped_cells


def is_cell_empty(cropped_cell, threshold=1):
    """
    Determine if a cell is empty based on the average pixel intensity.

    Parameters:
    - cell: Cropped cell image as a numpy array.
    - threshold: Pixel intensity threshold to consider the cell as empty.

    Returns:
    - True if the cell is considered empty, False otherwise.
    """
    return np.mean(cropped_cell) < threshold

def preprocess_for_mnist(cell):
    # Assuming cell has been optionally cropped already
    resized = cv2.resize(cell, (28, 28), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
    scaled = gray / 255.0
    return np.expand_dims(scaled, axis=0)  # Correct shape should be (1, 28, 28)

def preprocess_image_for_mnist(image):
    # Assuming `image` is your input image to be processed
    
    # Resize to 28x28
    processed_image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Convert to grayscale
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to mimic MNIST's binary-like images
    _, processed_image = cv2.threshold(processed_image, 127, 255, cv2.THRESH_BINARY)
    
    # Normalize pixel values to [0, 1] as done with MNIST
    processed_image = processed_image / 255.0
    
    # Expand dimensions to match the model's input shape (1, 28, 28, 1)
    processed_image = np.expand_dims(processed_image, axis=-1)
    processed_image = np.expand_dims(processed_image, axis=0)
    
    return processed_image
    

def recognize_digit(cell, model):
    preprocessed = preprocess_for_mnist(cell)
    prediction = model.predict(preprocessed, verbose =0)
    return np.argmax(prediction)  # Return the digit with the highest probability

def recognize_digits(cropped_cells, model, empty_cells_indices):
    """
    Recognize digits within each cell, populating the Sudoku matrix, using a list of empty cell indices.

    Parameters:
    - cropped_cells: A 9x9 grid (list of lists) of cropped cell images.
    - model: The trained MNIST model for digit recognition.
    - empty_cells_indices: A list of tuples indicating the indices of empty cells.

    Returns:
    - A 9x9 Sudoku matrix populated with recognized digits or 0 for empty cells.
    """
    # Initialize the sudoku matrix with placeholders to indicate non-reviewed cells
    sudoku_matrix = [[-1 for _ in range(9)] for _ in range(9)]

    for i, row in enumerate(cropped_cells):
        for j, cell in enumerate(row):
            if (i, j) in empty_cells_indices:
                sudoku_matrix[i][j] = '█'  # Mark cell as empty
            else:
                digit = recognize_digit(cell, model)
                sudoku_matrix[i][j] = digit  # Update with recognized digit
    return sudoku_matrix

def print_sudoku_matrix(sudoku_matrix):
    print("Sudoku Grid:")
    for i, row in enumerate(sudoku_matrix):
        if i % 3 == 0 and i != 0:  # Add a horizontal separator line between each 3 rows
            print("- - - - - - - - - - -")
        
        for j, num in enumerate(row):
            if j % 3 == 0 and j != 0:  # Add a vertical separator between each 3 columns
                print("| ", end="")
            
            # Adjust for None values (assuming empty cells are represented by None)
            cell_value = num if num is not None else ' '
            if j == 8:  # If it's the last number in the row, move to the next line after printing
                print(cell_value)
            else:  # Else, stay on the same line
                print(str(cell_value) + " ", end="")

                
def display_sudoku_solution(sudoku_matrix):
    print("Initial Sudoku Grid:")
    print_sudoku_matrix(sudoku_matrix)  # Print the initial state of the Sudoku grid
    print('\n\nSolving...\n')

    if solve_sudoku(sudoku_matrix):
        print("Solved Sudoku Grid:")
        print_sudoku_matrix(sudoku_matrix)  # Print the solved Sudoku grid
    else:
        print("No solution exists.")

def is_valid_move(board, row, col, num):
    # Check if num is not in the given row
    for x in range(9):
        if board[row][x] == num:
            return False

    # Check if num is not in the given column
    for x in range(9):
        if board[x][col] == num:
            return False

    # Check if num is not in the 3x3 subgrid
    start_row = row - row % 3
    start_col = col - col % 3
    for i in range(3):
        for j in range(3):
            if board[i + start_row][j + start_col] == num:
                return False

    return True

def solve_sudoku(board):
    empty = find_empty_location(board)
    if not empty:
        return True  # Puzzle solved
    row, col = empty

    for num in range(1, 10):  # Try all numbers from 1 to 9
        if is_valid_move(board, row, col, num):
            board[row][col] = num  # Place num

            if solve_sudoku(board):  # Continue with this placement
                return True

            board[row][col] = 0  # Undo & try again

    return False  # Trigger backtracking

def find_empty_location(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0 or board[i][j] == '█':  # Adjust this condition based on how you represent empty cells
                return (i, j)
    return None

def main():
    # Step 1: Capture the Sudoku puzzle screenshot
    screenshot = acquire_image()
    visualize_image(screenshot, "Original Screenshot")

    # Step 2: Preprocess the image to facilitate grid detection + visualizatoin
    preprocessed_image = preprocess_image(screenshot)
    visualize_image(preprocessed_image, "Preprocessed Image")

    # Step 3: Detect the Sudoku grid and extract individual cells + visualization
    cells = detect_sudoku_grid(preprocessed_image)
    if not cells:
        print("Failed to detect Sudoku grid or extract cells.")
        return
    visualize_cells_grid(cells, "Extracted Cells")
    
    # Step 4: Crop the cells to remove borders + visualization
    cropped_cells = crop_cells_grid(cells, border_size=6)
    visualize_cells_grid(cropped_cells, "Cropped Cells")

    # Step 5: Check thresholding for empty cells and capture empty cell indices
    empty_cells_indices = print_cell_grid_status(cropped_cells, threshold=1)

    # Step 6: Preprocess the cropped cells for MNIST (visualization step included)
    visualize_cells_grid_for_mnist(cropped_cells)

    # Path to the pre-trained digit recognition model
    model_path = 'my_mnist_model.keras'
    model = load_model(model_path)
    
    # Step 7: Recognize digits within each cell to populate the Sudoku matrix
    # Now passing the empty_cells_indices to recognize_digits
    sudoku_matrix = recognize_digits(cropped_cells, model, empty_cells_indices)

    #print_sudoku_matrix(sudoku_matrix)

    display_sudoku_solution(sudoku_matrix)

# Run the main program
if __name__ == "__main__":
    main()
