using MagicEntanglement
using Documenter

DocMeta.setdocmeta!(MagicEntanglement, :DocTestSetup, :(using MagicEntanglement); recursive=true)

makedocs(;
    modules=[MagicEntanglement],
    authors="Zhaohui Zhi",
    sitename="MagicEntanglement.jl",
    format=Documenter.HTML(;
        canonical="https://zzh-cycling.github.io/MagicEntanglement.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/zzh-cycling/MagicEntanglement.jl",
    devbranch="main",
)
