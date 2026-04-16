import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

class NetConverter:
    """
    Converts React Canvas nodes and edges into SUMO XML formats 
    and runs netconvert to generate a .net.xml file.
    """
    
    def __init__(self, output_dir: str = "simulation/sumo_configs/custom_maps"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def convert_to_sumo(
        self, 
        nodes: List[Dict], 
        edges: List[Dict], 
        map_name: str = "custom"
    ) -> Dict[str, str]:
        """
        Takes canvas data and produces .nod.xml, .edg.xml, and runs netconvert.
        
        nodes: list of {id, x, y, type}
        edges: list of {id, from, to, lanes, speed}
        """
        nod_file = self.output_dir / f"{map_name}.nod.xml"
        edg_file = self.output_dir / f"{map_name}.edg.xml"
        net_file = self.output_dir / f"{map_name}.net.xml"

        # 1. Generate Nodes XML
        node_root = ET.Element("nodes")
        for n in nodes:
            ET.SubElement(node_root, "node", {
                "id": str(n["id"]),
                "x": str(n["x"]),
                "y": str(n["y"]),
                "type": n.get("type", "priority")
            })
        
        self._write_xml(node_root, nod_file)

        # 2. Generate Edges XML
        edge_root = ET.Element("edges")
        for e in edges:
            ET.SubElement(edge_root, "edge", {
                "id": str(e["id"]),
                "from": str(e["from"]),
                "to": str(e["to"]),
                "numLanes": str(e.get("lanes", 1)),
                "speed": str(e.get("speed", 13.89)) # Default 50km/h in m/s
            })
        
        self._write_xml(edge_root, edg_file)

        # 3. Run netconvert
        try:
            cmd = [
                "netconvert",
                "--node-files", str(nod_file),
                "--edge-files", str(edg_file),
                "--output-file", str(net_file)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return {
                "success": "true",
                "net_file": str(net_file),
                "log": result.stdout
            }
        except subprocess.CalledProcessError as e:
            return {
                "success": "false",
                "error": e.stderr
            }
        except FileNotFoundError:
            return {
                "success": "false",
                "error": "netconvert not found in system PATH."
            }

    def _write_xml(self, root: ET.Element, file_path: Path):
        tree = ET.ElementTree(root)
        ET.indent(tree, space="    ", level=0)
        tree.write(file_path, encoding="utf-8", xml_declaration=True)
