#!/usr/bin/env python3
"""
Fix common linting issues for flake8
"""

import os
import re

def fix_file(filepath):
    """Fix linting issues in a Python file."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Remove trailing whitespace
        line = line.rstrip() + '\n' if line.strip() else line.rstrip() + '\n'
        
        # Check if this is a function/class definition
        if i > 0 and (line.strip().startswith('def ') or 
                      line.strip().startswith('class ') or
                      line.strip().startswith('@') or
                      line.strip().startswith('async def ')):
            # Count blank lines before this line
            blank_count = 0
            j = i - 1
            while j >= 0 and lines[j].strip() == '':
                blank_count += 1
                j -= 1
            
            # If inside a file (not at the beginning) and not inside a class
            if j >= 0:
                # Check if previous non-blank line is import-related
                prev_line = lines[j].strip()
                is_after_import = prev_line.startswith('import ') or prev_line.startswith('from ')
                
                # Need 2 blank lines for top-level definitions
                if blank_count < 2 and not lines[j].strip().startswith(' '):
                    # Add blank lines
                    while blank_count < 2:
                        fixed_lines.append('\n')
                        blank_count += 1
        
        fixed_lines.append(line)
        i += 1
    
    # Write back
    with open(filepath, 'w') as f:
        f.writelines(fixed_lines)
    
    print(f"Fixed: {filepath}")

def main():
    """Fix linting issues in backend files."""
    files_to_fix = [
        'backend/app/dependencies.py',
        'backend/app/main.py',
        'backend/app/middleware.py',
        'backend/app/state.py',
        'backend/config/settings.py'
    ]
    
    for filepath in files_to_fix:
        if os.path.exists(filepath):
            fix_file(filepath)
        else:
            print(f"File not found: {filepath}")

if __name__ == "__main__":
    main()