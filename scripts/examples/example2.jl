using SketchySVD
using TestImages, ImageIO
using LinearAlgebra

# Load image (or create synthetic)
img = Float64.(Gray.(testimage("cameraman")))
m, n = size(img)

# Compress with different ranks
ranks = [5, 10, 20, 50, 100]
compressed_images = []

for r in ranks
    U, S, V = rsvd(img, r)
    img_approx = U * Diagonal(S) * V'
    
    error = norm(img - img_approx) / norm(img)
    compression_ratio = (m * n) / (r * (m + n + 1))
    
    println("Rank $r: Error = $(round(error, digits=4)), Compression = $(round(compression_ratio, digits=2))x")
    push!(compressed_images, img_approx)
end

# Visualize results
using Plots
plot(
    plot(Gray.(img), title="Original"),
    plot(Gray.(compressed_images[1]), title="Rank 5"),
    plot(Gray.(compressed_images[3]), title="Rank 20"),
    plot(Gray.(compressed_images[5]), title="Rank 100"),
    layout=(2,2)
)