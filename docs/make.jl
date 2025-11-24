using Documenter
using DocumenterCitations
using SketchySVD

ENV["JULIA_DEBUG"] = "Documenter"
DocMeta.setdocmeta!(SketchySVD, :DocTestSetup, :(using SketchySVD); recursive=true)

PAGES = [
    "Home" => "index.md",
    "Mathematical Background" => [
        "Theory" => "theory.md",
        "Sketching Algorithms" => "sketching.md",
        "Randomized SVD" => "rsvd.md",
    ],
    "User Guide" => [
        "Getting Started" => "quickstart.md",
        "Dimension Reduction Maps" => "redux_maps.md",
        "Examples" => "examples.md",
    ],
    "API Reference" => "api.md",
    "References" => "paper.md",
]

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"))

makedocs(
    sitename = "SketchySVD",
    clean = true, doctest = false, linkcheck = false,
    authors = "Tomoki Koike <tkoike45@gmail.com>",
    repo = Remotes.GitHub("smallpondtom", "SketchySVD.jl"),
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        edit_link = "https://github.com/smallpondtom/SketchySVD.jl",
        assets=String[
            "assets/citations.css",
        ],
        # analytics = "G-B2FEJZ9J99",
    ),
    modules = [SketchySVD,],
    pages = PAGES,
    plugins=[bib],
)

deploydocs(
    repo = "github.com/smallpondtom/SketchySVD.jl",
    target = "build",
    branch = "gh-pages",
    devbranch = "main",
    push_preview = true,
    # Add other deployment options as needed
)