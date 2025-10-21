import PyInstaller.__main__

PyInstaller.__main__.run([
    'rag_excel_processor_gui.py',  # Main script to package
    '--windowed', 
    '--noconsole',  # No console window
    # '--onefile',  # Removed for macOS compatibility with --windowed
    # '--icon=Logo.jpeg',  # Path to icon file
    '--name=Offline AI bot',  # Name of the executable
])

