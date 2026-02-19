# ============================================================
# utils/config_loader.py
# ============================================================
# PURPOSE:
#   This utility module is responsible for loading and validating
#   the master configuration file (config.yaml). It acts as the
#   single source of truth for all settings across the application.
#
# WHY WE NEED THIS:
#   Instead of hardcoding values like API keys, model names, or
#   search settings inside each agent/tool file, we centralize
#   everything in config.yaml. This makes the project:
#     - Easy to configure (change one file, affects everything)
#     - Clean and maintainable
#     - Professional (industry standard practice)
#
# HOW IT WORKS:
#   1. Reads config.yaml using PyYAML
#   2. Validates that required fields (like API key) are present
#   3. Returns a Python dictionary that all modules can use
#
# USAGE:
#   from utils.config_loader import load_config
#   config = load_config()
#   api_key = config['api']['groq_api_key']
# ============================================================

import yaml          # PyYAML: reads .yaml files into Python dicts
import os            # For file path operations
import sys           # For exiting on critical errors


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load and validate the master configuration from a YAML file.

    WHY THIS FUNCTION EXISTS:
        Every agent, tool, and the UI needs access to settings like
        the API key, model name, search parameters, etc. This function
        provides a single, reliable way to load all those settings.
        It also validates the config so you get a clear error message
        if something is missing, rather than a cryptic crash later.

    HOW IT WORKS:
        1. Builds the absolute path to config.yaml
        2. Opens and parses the YAML file into a Python dictionary
        3. Validates that the Groq API key has been set
        4. Returns the complete configuration dictionary

    Args:
        config_path (str): Relative or absolute path to the config YAML file.
                           Defaults to "config.yaml" in the project root.

    Returns:
        dict: A nested dictionary containing all configuration settings.
              Structure mirrors the YAML file structure.
              Example:
                {
                  'api': {'groq_api_key': 'gsk_...'},
                  'llm': {'model': 'llama-3.1-70b-versatile', ...},
                  'search': {'max_results': 3, ...},
                  ...
                }

    Raises:
        FileNotFoundError: If config.yaml doesn't exist at the given path.
        ValueError: If the Groq API key is missing or still set to placeholder.
        yaml.YAMLError: If the YAML file has syntax errors.

    Example:
        >>> config = load_config()
        >>> print(config['llm']['model'])
        'llama-3.1-70b-versatile'
    """

    # ----------------------------------------------------------------
    # Step 1: Resolve the absolute path to config.yaml
    # We use the directory of this script as the base, then go up one
    # level to reach the project root where config.yaml lives.
    # ----------------------------------------------------------------
    # Get the directory where this config_loader.py file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Go up one level from utils/ to reach the project root
    project_root = os.path.dirname(current_dir)

    # Build the full path to config.yaml
    full_config_path = os.path.join(project_root, config_path)

    # ----------------------------------------------------------------
    # Step 2: Check if the config file actually exists
    # Give a helpful error message if it doesn't
    # ----------------------------------------------------------------
    if not os.path.exists(full_config_path):
        raise FileNotFoundError(
            f"\nERROR: Configuration file not found at: {full_config_path}\n"
            f"   Please make sure 'config.yaml' exists in the project root.\n"
            f"   Project root detected as: {project_root}"
        )

    # ----------------------------------------------------------------
    # Step 3: Open and parse the YAML file
    # yaml.safe_load() converts YAML into a Python dictionary safely
    # (safe_load prevents execution of arbitrary Python code in YAML)
    # ----------------------------------------------------------------
    try:
        with open(full_config_path, "r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    except yaml.YAMLError as e:
        # If the YAML has syntax errors, give a clear error message
        raise yaml.YAMLError(
            f"\nERROR: Error parsing config.yaml. Check for syntax errors.\n"
            f"   YAML Error: {e}"
        )

    # ----------------------------------------------------------------
    # Step 4: Validate that the Groq API key has been set
    # The placeholder value "your_groq_api_key_here" means the user
    # hasn't replaced it with their actual key yet.
    # ----------------------------------------------------------------
    validate_config(config)

    return config


def validate_config(config: dict) -> None:
    """
    Validate that all required configuration fields are properly set.

    WHY THIS FUNCTION EXISTS:
        Without validation, if the user forgets to set their API key,
        the app would crash with a confusing authentication error deep
        inside the LangChain/Groq code. This function catches common
        mistakes early and gives clear, actionable error messages.

    WHAT IT CHECKS:
        1. The 'api' section exists in config
        2. The 'groq_api_key' field exists
        3. The API key is not the placeholder value
        4. The API key is not empty

    Args:
        config (dict): The configuration dictionary loaded from config.yaml.

    Returns:
        None

    Raises:
        ValueError: If any required field is missing or invalid.

    Example:
        >>> validate_config({'api': {'groq_api_key': 'gsk_real_key'}})
        # No error raised — config is valid
        >>> validate_config({'api': {'groq_api_key': 'your_groq_api_key_here'}})
        # Raises ValueError with helpful message
    """

    # Check that the 'api' section exists in the config
    if "api" not in config:
        raise ValueError(
            "\nERROR: Missing 'api' section in config.yaml.\n"
            "   Please add:\n"
            "   api:\n"
            "     groq_api_key: 'your_actual_key_here'"
        )

    # Check that 'groq_api_key' exists within the 'api' section
    if "groq_api_key" not in config["api"]:
        raise ValueError(
            "\nERROR: Missing 'groq_api_key' in config.yaml under 'api' section.\n"
            "   Please add your Groq API key:\n"
            "   api:\n"
            "     groq_api_key: 'gsk_your_actual_key_here'"
        )

    api_key = config["api"]["groq_api_key"]

    # Check that the API key is not the placeholder value
    if api_key == "your_groq_api_key_here" or not api_key:
        raise ValueError(
            "\nERROR: Groq API key not set in config.yaml!\n"
            "   Please replace 'your_groq_api_key_here' with your actual key.\n"
            "   Get your free API key at: https://console.groq.com\n"
            "   Then update config.yaml:\n"
            "   api:\n"
            "     groq_api_key: 'gsk_your_actual_key_here'"
        )


def get_nested(config: dict, *keys, default=None):
    """
    Safely retrieve a nested value from the configuration dictionary.

    WHY THIS FUNCTION EXISTS:
        When accessing deeply nested config values like
        config['agents']['planner']['num_subquestions'], if any
        intermediate key is missing, Python raises a KeyError.
        This helper function safely navigates nested dicts and
        returns a default value if any key is missing.

    HOW IT WORKS:
        Iterates through the keys one by one, going deeper into
        the nested dictionary. If any key doesn't exist, returns
        the default value instead of crashing.

    Args:
        config (dict): The configuration dictionary.
        *keys: Variable number of keys to navigate the nested structure.
               Example: get_nested(config, 'agents', 'planner', 'num_subquestions')
        default: Value to return if any key is not found. Defaults to None.

    Returns:
        The value at the nested key path, or `default` if not found.

    Example:
        >>> config = {'agents': {'planner': {'num_subquestions': 5}}}
        >>> get_nested(config, 'agents', 'planner', 'num_subquestions')
        5
        >>> get_nested(config, 'agents', 'planner', 'missing_key', default=3)
        3
    """
    # Start at the top level of the config dictionary
    current = config

    # Navigate through each key one by one
    for key in keys:
        # If current level is a dict and the key exists, go deeper
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            # Key not found — return the default value
            return default

    return current
