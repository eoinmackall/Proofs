"""proofs — distributed proof construction, coordinated through git.

One command, eight subcommands (see cli.py):

    proofs config KEY [VALUE]
    proofs parse DIR | run DIR | verify PATH lemma_id [--full]
    proofs repair path_to_refutation | prune DIR
    proofs status PATH lemma_id | export PATH [lemma_id]
"""

__version__ = "0.1.0"
