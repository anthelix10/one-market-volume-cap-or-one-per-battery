ANALYSIS SCRIPTS
================

Run the commands from the repository root. Before running the numerical checks,
copy the inputs/ and paper_of_record/ folders from the Zenodo package into the
repository. Install the Python packages with:

    python -m pip install -r requirements.txt

Scripts
-------

run_paper_of_record.py
    Runs the main analysis. Use --validate-stored to check the saved results or
    --rebuild-all to rebuild them.

run_scope_extensions.py
    Runs the scope and allowance extensions. Use --validate-stored for the saved
    results or --rebuild to rerun the extension analysis.

run_cost_feasibility_extensions.py
    Runs the cost and physical-feasibility extensions. Use --validate-stored to
    check the saved results or --rebuild-all to rebuild them.

rebuild_phi_calibration.py
    Handles the calibration analysis. Use --validate-stored to check the saved
    calibration or --write to rebuild it.

run_partition_extensions.py
    Runs the partition analysis. Use --validate to check the saved results or
    --rebuild to rerun it.

make_article_figures.py
    Makes the article and supplementary figures. It does not need command-line
    options.

verify_inputs_from_raw_archive.py
    Compares the processed MID, PN and B1610 inputs with the separate raw-response
    ZIP from Zenodo. Pass the location of that ZIP with --raw-archive.
