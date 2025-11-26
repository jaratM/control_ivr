import os
import argparse
from pipeline.orchestrator import PipelineOrchestrator
from loguru import logger

def main():
    parser = argparse.ArgumentParser(description="Audio Compliance Verification Pipeline")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing audio files")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory not found: {args.input_dir}")
        return

    orchestrator = PipelineOrchestrator(args.config)
    results = orchestrator.run(args.input_dir)
    
    # Output results summary
    compliant_count = sum(1 for r in results if r.is_compliant)
    logger.info(f"Summary: {compliant_count}/{len(results)} files compliant.")
    
    for res in results:
        if not res.is_compliant:
            logger.warning(f"Non-compliant file {res.file_id}: {res.issues}")

if __name__ == "__main__":
    main()

