"""proofs — distributed proof construction, coordinated through git.

One command, thirteen subcommands (see cli/main.py):

    proofs config KEY [VALUE] | init [DIR] | new NAME
    proofs parse DIR | run DIR | verify DIR LEMMA_ID [--full]
    proofs repair path_to_refutation | prune DIR | migrate [DIR]
    proofs status DIR LEMMA_ID | export DIR [LEMMA_ID] -o OUT_DIR
    proofs limits DIR | tools lock DIR | tools check DIR
"""

__version__ = "0.2.0"
