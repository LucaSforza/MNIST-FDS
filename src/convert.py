from typing import Any
import json
import os
from omegaconf import OmegaConf

import ast

# 1. Definiamo un dizionario che mappa il nome del tipo alla funzione di conversione
TYPE_MAP = {
    "int": int,
    "integer": int,
    "float": float,
    "str": str,
    "string": str,
    # Funzione lambda (anonima) per gestire correttamente i booleani
    "bool": lambda x: str(x).lower() in ["true", "1", "t", "y", "yes"],
    "boolean": lambda x: str(x).lower() in ["true", "1", "t", "y", "yes"],
    # ast.literal_eval per valutare stringhe come "[1, 2]" in modo sicuro
    "list": ast.literal_eval,
    "tuple": ast.literal_eval,
    "dict": ast.literal_eval,
}


def parse_value(val_str: str, type_name: str):
    # Gestione dei nulli
    if str(val_str).lower() in ["none", "null", "undefined", ""]:
        return None

    # 2. Recuperiamo la funzione corretta dal dizionario.
    # Se il tipo non esiste, usiamo 'str' come fallback.
    converter_function = TYPE_MAP.get(type_name.lower(), str)

    try:
        # 3. Eseguiamo la funzione passandogli la stringa
        return converter_function(val_str)
    except (ValueError, SyntaxError, TypeError) as e:
        print(f"Impossibile convertire '{val_str}' in '{type_name}'. Errore: {e}")
        return val_str


def build_hydra_configs(json_path: str, output_dir: str = "cfg"):
    # Carica il JSON generato da Svelte
    with open(json_path, "r") as f:
        diagram: dict[str, Any] = json.load(f)
    sequence: list[Any] = diagram["network"]
    # Crea la struttura delle cartelle
    layer_dir = os.path.join(output_dir, "layer")
    net_dir = os.path.join(output_dir, "net")
    os.makedirs(layer_dir, exist_ok=True)
    os.makedirs(net_dir, exist_ok=True)

    net_defaults: list[dict[str, str]] = []

    sequence_ids: list[str] = []

    for layer in sequence:
        layer_id: str = layer["id"]
        sequence_ids.append(layer_id)

        # 1. Costruisci il dizionario per il singolo file YAML
        layer_conf_dict = {"_target_": layer["target"]}

        for param_name, param in layer["params"].items():
            layer_conf_dict[param_name] = parse_value(param["value"], param["type"])

        # Rimuovi eventuali chiavi con None se preferisci non stamparle in Hydra
        layer_conf_dict = {k: v for k, v in layer_conf_dict.items() if v is not None}

        # Salva il file YAML per questo specifico layer (es: cfg/layer/layer_0.yaml)
        layer_yaml_path = os.path.join(layer_dir, f"{layer_id}.yaml")
        OmegaConf.save(config=layer_conf_dict, f=layer_yaml_path)

        # 2. Aggiungi il riferimento per il file defaults della rete
        # Formato Hydra: { "layer/layer_0": layer_id }
        net_defaults.append({f"layer/{layer_id}": layer_id})

    # 3. Genera il file principale della rete (es: cfg/net/custom_sequence.yaml)
    net_config = OmegaConf.create(
        {
            "defaults": net_defaults,
            "sequence": sequence_ids,
            "name": "SequentialModel_from_UI",
        }
    )

    net_yaml_path = os.path.join(net_dir, "custom_sequence.yaml")
    OmegaConf.save(config=net_config, f=net_yaml_path)

    print(f"✅ Configurazione salvata con successo in '{output_dir}/'!")


if __name__ == "__main__":
    # Sostituisci col percorso in cui Svelte salva/scarica il JSON
    build_hydra_configs("model_export.json")
