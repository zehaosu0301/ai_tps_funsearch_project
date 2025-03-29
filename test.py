import os
import sys
import numpy as np
import threading
import time
import traceback
from typing import Any, Collection, Dict, Tuple
import json
from collections.abc import Sequence
from openai import OpenAI
import textwrap
import openai
# Add the project root to the path
sys.path.append('/ai_tps_funsearch_project/')

# Import FunSearch modules
from implementation import config as config_lib
from implementation import code_manipulation
from implementation import programs_database
from implementation import evaluator
from implementation import sampler
from implementation import funsearch

#-------------------------------------------------------------------------
# 1. LLM Implementation
#-------------------------------------------------------------------------
class CustomLLM(sampler.LLM):
    """OpenAI-based language model for TSP code generation."""

    def __init__(
        self, 
        samples_per_prompt: int, 
        api_key: str = None,
        base_url: str = "https://api.bltcy.ai/v1",
        model: str = "gpt-4o",
        max_tokens: int = 1500,
        temperature: float = 0.7,
        trim: bool = True
    ) -> None:
        super().__init__(samples_per_prompt)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._trim = trim
        self._api_key = api_key
        self._base_url = base_url
        
        # Add TSP-specific prompt guidance
        self._additional_prompt = (
            'Complete a different and more complex Python function for the TSP problem. '
            'Be creative and innovative in your approach to the traveling salesman problem. '
            'You can insert multiple if-else and for-loop in the code logic. '
            'Only output the Python code, no descriptions.'
        )
        
        # Initialize the OpenAI client with custom base URL
        if not api_key:
            raise ValueError("API key is required")
            
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def _draw_sample(self, prompt: str) -> str:
        """Returns a predicted continuation of `prompt` using API."""
        full_prompt = '\n'.join([prompt, self._additional_prompt])
        
        try:
            # Using the OpenAI client with custom base URL
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": "You are a Python expert focusing on optimization algorithms."},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                stop=["def ", "\n\n\n"]
            )
            
            result = response.choices[0].message.content
            
            # Trim the response if needed
            if self._trim:
                result = self._trim_preface_of_body(result)
                
            return result
        
        except Exception as e:
            print(f"Error in LLM API call: {e}")
            # Retry after a short delay
            time.sleep(2)
            return self._draw_sample(prompt)  # Recursive retry

    def _trim_preface_of_body(self, sample: str) -> str:
        """Trim the redundant descriptions/symbols/'def' declaration before the function body."""
        lines = sample.splitlines()
        func_body_lineno = 0
        find_def_declaration = False
        
        for lineno, line in enumerate(lines):
            # Find the first 'def' statement in the given code
            if line[:3] == 'def':
                func_body_lineno = lineno
                find_def_declaration = True
                break
                
        if find_def_declaration:
            code = ''
            for line in lines[func_body_lineno + 1:]:
                code += line + '\n'
            return code
            
        return sample

    def draw_samples(self, prompt: str) -> Collection[str]:
        """Returns multiple predicted continuations of `prompt`."""
        return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]

#-------------------------------------------------------------------------
# 2. Sandbox Implementation
#-------------------------------------------------------------------------
class CustomSandbox(evaluator.Sandbox):
    """Safe execution environment for TSP code evaluation."""

    def __init__(self, verbose=False, numba_accelerate=False, timeout_seconds=30):
        """
        Args:
            verbose: Print evaluation information
            numba_accelerate: Whether to use numba acceleration
            timeout_seconds: Maximum execution time allowed
        """
        self._verbose = verbose
        self._numba_accelerate = numba_accelerate
        self._timeout_seconds = timeout_seconds

    def run(
        self,
        program: str,
        function_to_run: str,
        test_input: Any,
        timeout_seconds: int = None,  # Use instance default if not provided
        function_to_evolve: str = None,  # Added for compatibility
        inputs: Any = None,  # Added for compatibility
        **kwargs
    ) -> Tuple[Any, bool]:
        """Safely execute the generated code within a timeout."""
        # Use instance timeout if not provided
        if timeout_seconds is None:
            timeout_seconds = self._timeout_seconds
            
        # For TSP dataset compatibility
        dataset = test_input
        if inputs is not None and isinstance(test_input, str):
            dataset = inputs[test_input]
            
        try:
            # Variables to store execution results
            result = [None]
            success = [False]
            exception_info = [None]
            
            # Function to execute the code
            def execute_code():
                try:
                    # Create a separate namespace for executing the code
                    namespace = {'np': np}  # Add numpy for TSP functions
                    
                    # Add numba acceleration if requested and function_to_evolve is provided
                    if self._numba_accelerate and function_to_evolve:
                        try:
                            import numba
                            # Find the function_to_evolve in program
                            import re
                            pattern = rf"def\s+{function_to_evolve}\s*\("
                            match = re.search(pattern, program)
                            if match:
                                # Add @numba.jit() decorator
                                program_lines = program.split('\n')
                                insert_pos = match.start()
                                line_num = program[:insert_pos].count('\n')
                                program_lines.insert(line_num, '@numba.jit(nopython=True)')
                                modified_program = '\n'.join(program_lines)
                                if self._verbose:
                                    print(f"Added numba acceleration to {function_to_evolve}")
                            else:
                                modified_program = program
                        except ImportError:
                            modified_program = program
                            if self._verbose:
                                print("Numba not available, skipping acceleration")
                    else:
                        modified_program = program
                    
                    # Execute the program in the namespace
                    exec(modified_program, namespace)
                    
                    # Check if the target function exists
                    if function_to_run not in namespace:
                        exception_info[0] = f"Function '{function_to_run}' not found in the program"
                        return
                    
                    # Record execution time
                    start_time = time.time()
                    
                    # Execute the function with the input
                    result[0] = namespace[function_to_run](dataset)
                    
                    execution_time = time.time() - start_time
                    if self._verbose:
                        print(f"Function executed in {execution_time:.4f} seconds")
                    
                    # Ensure result is a number
                    if not isinstance(result[0], (int, float)):
                        exception_info[0] = f"Result must be int or float, got {type(result[0])}"
                        return
                        
                    # Mark as successful
                    success[0] = True
                    
                except Exception as e:
                    exception_info[0] = f"Execution error: {str(e)}\n{traceback.format_exc()}"
                    if self._verbose:
                        print(exception_info[0])
            
            # Create and start the execution thread
            thread = threading.Thread(target=execute_code)
            thread.daemon = True
            thread.start()
            
            # Wait for the thread to complete or timeout
            thread.join(timeout_seconds)
            
            # Check if the thread is still running (timeout occurred)
            if thread.is_alive():
                if self._verbose:
                    print(f"Execution timed out after {timeout_seconds} seconds")
                return None, False
            
            # Check if there was an exception
            if exception_info[0]:
                if self._verbose:
                    print(exception_info[0])
                return None, False
            
            return result[0], success[0]
            
        except Exception as e:
            if self._verbose:
                print(f"Sandbox error: {str(e)}")
            return None, False

#-------------------------------------------------------------------------
# 3. FunSearch Integration
#-------------------------------------------------------------------------
class ClassConfig:
    """Configuration for classes used in FunSearch."""
    
    def __init__(self, llm_class=None, sandbox_class=None):
        self.llm_class = llm_class or CustomLLM
        self.sandbox_class = sandbox_class or CustomSandbox

def run_custom_funsearch(
    specification,
    inputs,
    api_key,
    base_url="https://api.bltcy.ai/v1",
    model="gpt-4o",
    max_tokens=1500,
    temperature=0.7,
    num_samples=1, 
    timeout_seconds=30,
    max_iterations=50,
    verbose=True,
    log_dir=None,
    use_numba=False
):
    """
    Custom function to run FunSearch with more control over the process.
    Unlike the original funsearch.main, this function:
    1. Limits the number of iterations (the original runs indefinitely)
    2. Allows custom API configuration
    3. Provides better error handling and reporting
    4. Supports custom timeout values
    
    Args:
        specification: The problem specification code
        inputs: The dataset of TSP instances
        api_key: API key for the language model
        base_url: Base URL for the API
        model: LLM model to use
        max_tokens: Maximum tokens for LLM generation
        temperature: Temperature for LLM sampling
        num_samples: Number of samples per prompt
        timeout_seconds: Maximum execution time per evaluation
        max_iterations: Maximum number of iterations
        verbose: Whether to print verbose information
        log_dir: Directory to save logs
        use_numba: Whether to use numba acceleration
    """
    # Create configuration with valid parameters
    # Note: We removed the invalid 'evaluate_timeout_seconds' parameter
    config = config_lib.Config(
        samples_per_prompt=num_samples
    )
    
    # Extract function names
    function_to_evolve, function_to_run = _extract_function_names(specification)
    
    # Parse the program
    template = code_manipulation.text_to_program(specification)
    
    # Create database
    database = programs_database.ProgramsDatabase(
        config.programs_database, template, function_to_evolve
    )
    
    # Create evaluators
    evaluators_list = []
    for _ in range(config.num_evaluators):
        eval_instance = evaluator.Evaluator(
            database,
            template,
            function_to_evolve,
            function_to_run,
            inputs,
            timeout_seconds=timeout_seconds  # Pass timeout directly to evaluator
        )
        # Replace with our custom sandbox
        eval_instance._sandbox = CustomSandbox(
            verbose=verbose, 
            numba_accelerate=use_numba,
            timeout_seconds=timeout_seconds
        )
        evaluators_list.append(eval_instance)
    
    # Send initial implementation for analysis
    initial = template.get_function(function_to_evolve).body
    evaluators_list[0].analyse(initial, island_id=None, version_generated=None)
    
    # Create samplers with our LLM implementation
    samplers_list = []
    for _ in range(config.num_samplers):
        sampler_instance = sampler.Sampler(
            database, evaluators_list, config.samples_per_prompt
        )
        # Replace the LLM with our implementation
        sampler_instance._llm = CustomLLM(
            config.samples_per_prompt,
            api_key=api_key,
            base_url=base_url,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature
        )
        samplers_list.append(sampler_instance)
    
    # Run FunSearch
    sample_count = 0
    try:
        print(f"Starting FunSearch with {max_iterations} iterations...")
        
        # Record best score
        best_score = float('-inf')
        best_program = None
        
        # Run the samplers (limited to max_iterations)
        for i in range(max_iterations):
            print(f"Iteration {i+1}/{max_iterations}")
            
            # Get a prompt
            prompt = database.get_prompt()
            
            # Sample from LLM
            samples = samplers_list[0]._llm.draw_samples(prompt.code)
            
            for sample in samples:
                # Choose an evaluator
                chosen_evaluator = evaluators_list[0]
                
                # Analyze the sample
                chosen_evaluator.analyse(
                    sample, prompt.island_id, prompt.version_generated
                )
                
                # Check current best score
                current_best = max(database._best_score_per_island)
                if current_best > best_score:
                    best_score = current_best
                    # Find the best program
                    best_island = database._best_score_per_island.index(best_score)
                    best_program = database._best_program_per_island[best_island]
                    print(f"New best score: {best_score}")
                
                sample_count += 1
                
        print(f"FunSearch completed after {sample_count} samples")
        print(f"Best score: {best_score}")
        
        if best_program:
            print("\nBest program:")
            print(str(best_program))
            
        return {
            "best_score": best_score,
            "best_program": best_program,
            "sample_count": sample_count
        }
        
    except Exception as e:
        print(f"Error during FunSearch: {e}")
        traceback.print_exc()
        return {
            "error": str(e),
            "sample_count": sample_count
        }

def _extract_function_names(specification: str) -> tuple[str, str]:
    """Returns the name of the function to evolve and of the function to run."""
    run_functions = list(
        code_manipulation.yield_decorated(specification, "funsearch", "run")
    )
    if len(run_functions) != 1:
        raise ValueError("Expected 1 function decorated with `@funsearch.run`.")
    evolve_functions = list(
        code_manipulation.yield_decorated(specification, "funsearch", "evolve")
    )
    if len(evolve_functions) != 1:
        raise ValueError("Expected 1 function decorated with `@funsearch.evolve`.")
    return evolve_functions[0], run_functions[0]

#-------------------------------------------------------------------------
# 4. TSP Data Initialization Functions
#-------------------------------------------------------------------------
import math

def get_matrix(cal_type, coords):
    """
    Calculate distance matrix for TSP instances based on different distance metrics.
    
    Args:
        cal_type: Type of distance calculation ('EUC_2D', 'ATT', 'CEIL_2D', 'GEO')
        coords: List of coordinates for each city
        
    Returns:
        Distance matrix as numpy array
    """
    n = len(coords)
    matrix = np.zeros((n, n), dtype=np.float32 if cal_type == "GEO" else np.int32)
    
    if cal_type == "EUC_2D":
        dist_func = lambda a, b: round(math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2))
    elif cal_type == "ATT":
        dist_func = lambda a, b: math.ceil(math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2) / 10)
    elif cal_type == "CEIL_2D":
        dist_func = lambda a, b: math.ceil(math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2))
    elif cal_type == "GEO":
        def dist_func(a, b):
            lat1, lon1 = math.radians(a[0]), math.radians(a[1])
            lat2, lon2 = math.radians(b[0]), math.radians(b[1])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
            return 6371.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))  # Earth radius 6371km
    else:
        return None

    for i in range(n):
        for j in range(i+1, n):
            dist = dist_func(coords[i], coords[j])
            matrix[i][j] = dist
            matrix[j][i] = dist

    return matrix

def prepare_tsp_datasets(tsp_data_list):
    """
    Prepare multiple TSP datasets from raw data.
    
    Args:
        tsp_data_list: List of tuples (calculation_type, coordinates)
        
    Returns:
        Dictionary of distance matrices indexed by instance names
    """
    matrices = {}
    for idx, (calc_type, coords) in enumerate(tsp_data_list):
        matrix = get_matrix(calc_type, coords)
        if matrix is not None:
            instance_name = f"tsp_instance_{idx}_{calc_type}"
            matrices[instance_name] = matrix
    return matrices

def load_tsp_from_file(filename):
    """
    Load TSP instance from TSPLIB format file.
    
    Args:
        filename: Path to the TSPLIB format file
        
    Returns:
        Tuple of (calculation_type, coordinates)
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    calc_type = None
    dimension = 0
    coords = []
    reading_coords = False
    
    for line in lines:
        line = line.strip()
        if line.startswith("EDGE_WEIGHT_TYPE"):
            calc_type = line.split(":")[1].strip()
            # Map TSPLIB format to our calculation types
            if calc_type == "EUC_2D":
                calc_type = "EUC_2D"
            elif calc_type == "ATT":
                calc_type = "ATT"
            elif calc_type == "GEO":
                calc_type = "GEO"
            else:
                calc_type = "EUC_2D"  # Default
        elif line.startswith("DIMENSION"):
            dimension = int(line.split(":")[1].strip())
        elif line == "NODE_COORD_SECTION":
            reading_coords = True
        elif reading_coords and line != "EOF":
            parts = line.split()
            if len(parts) >= 3:
                # Format: node_id x y (node_id is 1-based, we make it 0-based)
                coords.append((float(parts[1]), float(parts[2])))
        elif line == "EOF":
            break
    
    return calc_type, coords

# Example usage
if __name__ == "__main__":
    # Define your TSP problem specification

    
    specification = """
import numpy as np

def heuristic_tsp_solver(distances, priority_func):
    num_cities = len(distances)
    visited = [False] * num_cities
    current_city = 0
    visited[current_city] = True
    tour = [current_city]
    total_distance = 0

    while len(tour) < num_cities:
        # Get complex priorities for the next city based on the current city
        priorities = priority_func(current_city, distances, visited)
        # Mask priorities for visited cities to ensure they are not selected
        masked_priorities = np.where(visited, np.inf, priorities)
        # Select the next city with the highest priority (lowest cost)
        next_city = np.argmin(masked_priorities)
        visited[next_city] = True
        tour.append(next_city)
        total_distance += distances[current_city][next_city]
        current_city = next_city

    # Close the loop by returning to the starting city
    total_distance += distances[current_city][tour[0]]
    tour.append(tour[0])  # Optionally return to the starting city for visualization
    return tour, total_distance

@funsearch.run
def evaluate(distance_matrix):
    
    tour, total_distance = heuristic_tsp_solver(distance_matrix, priority)
    return -total_distance  # Negative sign to maximize in FunSearch

@funsearch.evolve
def priority(current_city, distances, visited):
   
    num_cities = len(distances)
    priorities = np.full(num_cities, np.inf)
    for city in range(num_cities):
        if not visited[city]:
            # Inverse of the distance (closer cities have higher priority)
            distance_priority = -distances[current_city][city]
            # Additional heuristic: dynamic cost based on some external condition or iteration dependent factor
            dynamic_cost = 1 / (1 + np.sum(distances[city]))  # Example: inverse of sum of distances from this city
            priorities[city] = distance_priority * dynamic_cost
    return priorities
"""
# Define your specification with indentation for readability

# Use textwrap.dedent to remove common leading whitespace
    
    
    # Sample TSP data: List of (calculation_type, coordinates)
    # Format: (calc_type, [(x1,y1), (x2,y2), ...])
    tsp_datas = [
        # Small example with 5 cities in EUC_2D format
        ("EUC_2D", [(0, 0), (10, 10), (20, 20), (30, 10), (15, 5)]),
        
        # Berlin52 example (first 10 cities) in EUC_2D format
        ("EUC_2D", [
            (565.0, 575.0), (25.0, 185.0), (345.0, 750.0), (945.0, 685.0), (845.0, 655.0),
            (880.0, 660.0), (25.0, 230.0), (525.0, 1000.0), (580.0, 1175.0), (650.0, 1130.0)
        ]),
        
        # Small GEO example with 5 cities (latitude, longitude)
        ("GEO", [(40.7128, -74.0060), (34.0522, -118.2437), (41.8781, -87.6298), 
                 (29.7604, -95.3698), (39.9526, -75.1652)])
    ]
    
    # Prepare dataset from our TSP instances
    dataset = prepare_tsp_datasets(tsp_datas)
    
    print(f"Prepared {len(dataset)} TSP instances for optimization")
    for name, matrix in dataset.items():
        print(f"  - {name}: {matrix.shape[0]} cities")
    
    # Run FunSearch with our custom function
    # Note: We changed 'run_funsearch' to 'run_custom_funsearch' to make the distinction clear
    results = run_custom_funsearch(
        specification=specification,
        inputs=dataset,
        api_key= "sk-RwjJjBq7VVTFvhv9352929B353Bb41D68d2f0959EcEe3b6f",
        base_url="https://api.bltcy.ai/v1",
        model="gpt-4o",
        max_iterations=10,
        verbose=True
    )
    
    print(f"Results: {results}")