"""proofs — distributed proof construction, coordinated through git.

One command, nine subcommands (see cli.py):

    proofs config KEY [VALUE] | new NAME
    proofs parse DIR | run DIR | verify PATH lemma_id [--full]
    proofs repair path_to_refutation | prune DIR
    proofs status PATH lemma_id | export PATH [lemma_id] -o OUT_DIR
"""

__version__ = "0.1.0"
