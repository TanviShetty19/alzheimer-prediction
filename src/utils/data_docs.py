import json
from typing import Dict

def generate_markdown_docs(metadata: Dict, output_path: str) -> None:
    """Generate Markdown documentation from metadata"""
    md_content = ["# Clinical Data Dictionary\n"]
    
    for col, specs in metadata["columns"].items():
        md_content.append(f"## `{col}`")
        md_content.append(f"- **Description**: {specs.get('description', 'Not documented')}")
        
        if "type" in specs:
            md_content.append(f"- **Type**: {specs['type']}")
            
        if "values" in specs:
            md_content.append(f"- **Categories**: {', '.join(map(str, specs['values']))}")
            
        if "range" in specs:
            md_content.append(f"- **Range**: {specs['range'][0]} to {specs['range'][1]}")
            
        md_content.append("")  # Empty line
    
    with open(output_path, 'w') as f:
        f.write("\n".join(md_content))