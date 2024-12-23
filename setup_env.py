# setup_env.py
import os
import subprocess
import sys
import venv

def create_and_setup_environment(base_path):
    # Create the project directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)
    
    # Create virtual environment
    env_path = os.path.join(base_path, 'venv')
    print(f"Creating virtual environment in {env_path}...")
    venv.create(env_path, with_pip=True)
    
    # Determine paths
    if sys.platform == 'win32':
        python_path = os.path.join(env_path, 'Scripts', 'python.exe')
        pip_path = os.path.join(env_path, 'Scripts', 'pip.exe')
    else:
        python_path = os.path.join(env_path, 'bin', 'python')
        pip_path = os.path.join(env_path, 'bin', 'pip')
    
    # Upgrade pip
    print("Upgrading pip...")
    subprocess.run([python_path, '-m', 'pip', 'install', '--upgrade', 'pip'])
    
    # Create requirements.txt
    requirements = '''PyQt5==5.15.9
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
librosa==0.10.1
soundfile==0.12.1
joblib==1.3.1'''

    req_path = os.path.join(base_path, 'requirements.txt')
    with open(req_path, 'w') as f:
        f.write(requirements)
    
    # Install requirements
    print("Installing requirements...")
    subprocess.run([pip_path, 'install', '-r', req_path])
    
    print("\nEnvironment setup complete!")
    print("\nTo activate the virtual environment:")
    if sys.platform == 'win32':
        print(f"    {env_path}\\Scripts\\activate.bat")
    else:
        print(f"    source {env_path}/bin/activate")

if __name__ == "__main__":
    base_path = r"C:\Users\karsd\Documents\Python\audio_analyser"
    create_and_setup_environment(base_path)
