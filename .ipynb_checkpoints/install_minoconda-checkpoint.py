import os
import subprocess

def run_command(command):
    """Execute a shell command and print output."""
    print(f"Running: {command}")
    process = subprocess.run(command, shell=True, capture_output=True, text=True)
    if process.returncode == 0:
        print(process.stdout)
    else:
        print(f"Error: {process.stderr}")
        exit(1)

def install_miniconda():
    """Download and install Miniconda on Ubuntu."""
    print("Starting Miniconda installation...")

    # Step 1: Download Miniconda Installer
    miniconda_url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    run_command(f"wget {miniconda_url} -O Miniconda3-latest-Linux-x86_64.sh")

    # Step 2: Run the Installer
    run_command("bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3")

    # Step 3: Initialize Conda
    run_command("$HOME/miniconda3/bin/conda init")

    # Step 4: Update .bashrc
    run_command("echo 'export PATH=\"$HOME/miniconda3/bin:$PATH\"' >> ~/.bashrc")

    # Step 5: Reload Shell
    run_command("source ~/.bashrc")

    # Step 6: Verify Installation
    run_command("conda --version")

    print("\n🎉 Miniconda installation complete! Restart your shell or run `source ~/.bashrc` to use Conda.")

if __name__ == "__main__":
    install_miniconda()
