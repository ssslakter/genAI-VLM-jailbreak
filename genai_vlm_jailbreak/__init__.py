def main() -> None:
    # Imported here: the train environment has no Cairo or imageio, and it only needs frames and score.
    from genai_vlm_jailbreak.generator import main as generate_dataset

    generate_dataset()
