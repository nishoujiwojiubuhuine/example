"""Perform PCA with pure Python math and output an SVG visualization.

The script builds a small 2D dataset, mean-centers it, computes a
singular value decomposition (SVD) without external numeric libraries,
and produces an SVG plot that highlights the principal component axes.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Sequence, Tuple

Matrix = List[List[float]]
Vector = List[float]


def mean_vector(matrix: Sequence[Sequence[float]]) -> Vector:
    columns = len(matrix[0])
    n_rows = len(matrix)
    return [sum(row[i] for row in matrix) / n_rows for i in range(columns)]


def center_matrix(matrix: Sequence[Sequence[float]], mean: Sequence[float]) -> Matrix:
    return [[value - mean[i] for i, value in enumerate(row)] for row in matrix]


def transpose(matrix: Sequence[Sequence[float]]) -> Matrix:
    return [list(col) for col in zip(*matrix)]


def gram_matrix(matrix: Sequence[Sequence[float]]) -> Matrix:
    # Computes X^T X for the given matrix X.
    rows = len(matrix)
    cols = len(matrix[0])
    result = [[0.0 for _ in range(cols)] for _ in range(cols)]
    for i in range(cols):
        for j in range(i, cols):
            value = sum(matrix[row][i] * matrix[row][j] for row in range(rows))
            result[i][j] = result[j][i] = value
    return result


def eigen_decomposition_2x2(matrix_2x2: Sequence[Sequence[float]]) -> Tuple[List[float], List[Vector]]:
    a = matrix_2x2[0][0]
    b = matrix_2x2[0][1]
    d = matrix_2x2[1][1]

    trace = a + d
    det = a * d - b * b
    discriminant = max(trace * trace - 4.0 * det, 0.0)
    sqrt_discriminant = math.sqrt(discriminant)

    lambda1 = (trace + sqrt_discriminant) / 2.0
    lambda2 = (trace - sqrt_discriminant) / 2.0

    eigenvalues = [lambda1, lambda2]

    if abs(b) > 1e-12:
        v1 = [lambda1 - d, b]
        v2 = [lambda2 - d, b]
    else:
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]

    def normalize(vector: Vector) -> Vector:
        norm = math.hypot(vector[0], vector[1])
        if norm < 1e-12:
            return [1.0, 0.0]
        return [component / norm for component in vector]

    v1 = normalize(v1)
    # Ensure the second eigenvector is orthogonal to the first.
    v2 = normalize([-v1[1], v1[0]])

    eigenvectors = [v1, v2]
    paired = sorted(zip(eigenvalues, eigenvectors), key=lambda item: item[0], reverse=True)
    eigenvalues_sorted, eigenvectors_sorted = zip(*paired)
    return list(eigenvalues_sorted), [list(vec) for vec in eigenvectors_sorted]


def matrix_vector_product(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> Vector:
    return [sum(row[i] * vector[i] for i in range(len(vector))) for row in matrix]


def matrix_multiply(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> Matrix:
    rows_a, cols_a = len(a), len(a[0])
    rows_b, cols_b = len(b), len(b[0])
    if cols_a != rows_b:
        raise ValueError("Incompatible shapes for matrix multiplication")

    result = [[0.0 for _ in range(cols_b)] for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            result[i][j] = sum(a[i][k] * b[k][j] for k in range(cols_a))
    return result


def make_diagonal_matrix(values: Sequence[float]) -> Matrix:
    size = len(values)
    diagonal = [[0.0 for _ in range(size)] for _ in range(size)]
    for i, value in enumerate(values):
        diagonal[i][i] = value
    return diagonal


def format_matrix(matrix: Sequence[Sequence[float]]) -> str:
    return "\n".join(
        "[" + ", ".join(f"{value: .6f}" for value in row) + "]" for row in matrix
    )


def create_svg(
    samples: Sequence[Sequence[float]],
    mean: Sequence[float],
    principal_vectors: Sequence[Tuple[Sequence[float], str, str]],
    output_path: Path,
) -> None:
    width, height = 600, 600
    margin = 50

    points = list(samples)
    axis_endpoints = [
        [mean[0] + vector[0], mean[1] + vector[1]] for vector, _, _ in principal_vectors
    ]

    xs = [p[0] for p in points] + [mean[0]] + [pt[0] for pt in axis_endpoints]
    ys = [p[1] for p in points] + [mean[1]] + [pt[1] for pt in axis_endpoints]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)

    def to_svg_coords(x: float, y: float) -> Tuple[float, float]:
        px = margin + (x - min_x) / span_x * (width - 2 * margin)
        py = height - (margin + (y - min_y) / span_y * (height - 2 * margin))
        return px, py

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs>'
        '<marker id="arrow-1f77b4" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#1f77b4" />'
        '</marker>'
        '<marker id="arrow-ff7f0e" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#ff7f0e" />'
        '</marker>'
        '</defs>',
        '<rect width="100%" height="100%" fill="#ffffff" />',
        '<g stroke="#cccccc" stroke-dasharray="4,4">',
    ]

    for i in range(6):
        x = margin + i * (width - 2 * margin) / 5
        y = margin + i * (height - 2 * margin) / 5
        lines.append(f'<line x1="{x:.2f}" y1="{margin}" x2="{x:.2f}" y2="{height - margin}" />')
        lines.append(f'<line x1="{margin}" y1="{y:.2f}" x2="{width - margin}" y2="{y:.2f}" />')
    lines.append('</g>')

    lines.append('<g fill="#1f77b4" stroke="#0d3c61" stroke-width="1">')
    for x, y in points:
        px, py = to_svg_coords(x, y)
        lines.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="5" />')
    lines.append('</g>')

    mean_px, mean_py = to_svg_coords(mean[0], mean[1])
    lines.append('<g stroke="#000000" stroke-width="2">')
    lines.append(
        f'<line x1="{mean_px - 6:.2f}" y1="{mean_py - 6:.2f}" x2="{mean_px + 6:.2f}" y2="{mean_py + 6:.2f}" />'
    )
    lines.append(
        f'<line x1="{mean_px - 6:.2f}" y1="{mean_py + 6:.2f}" x2="{mean_px + 6:.2f}" y2="{mean_py - 6:.2f}" />'
    )
    lines.append('</g>')

    for vector, label, color in principal_vectors:
        end_x, end_y = mean[0] + vector[0], mean[1] + vector[1]
        end_px, end_py = to_svg_coords(end_x, end_y)
        marker_id = f"arrow-{color.strip('#')}"
        lines.append(
            f'<line x1="{mean_px:.2f}" y1="{mean_py:.2f}" x2="{end_px:.2f}" y2="{end_py:.2f}" '
            f'stroke="{color}" stroke-width="3" marker-end="url(#{marker_id})" />'
        )
        lines.append(
            f'<text x="{end_px:.2f}" y="{end_py - 10:.2f}" fill="{color}" font-size="18" text-anchor="middle">{label}</text>'
        )

    lines.append('</svg>')

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    # Construct the 2D dataset provided by the user (rows: samples, columns: features).
    raw_data = """
    0.808620 0.112335
    0.837497 0.295085
    0.438965 0.138758
    -0.414321 -0.171968
    0.317234 -0.036543
    1.157469 0.389024
    0.866828 0.433538
    1.530177 0.490791
    -1.436181 -0.460006
    0.058143 0.038909
    0.926981 0.473687
    -0.365502 -0.071732
    1.087869 0.418758
    -0.617016 -0.257694
    -0.317234 -0.012673
    -0.896645 -0.441484
    1.482131 0.488155
    0.389837 0.247574
    -0.697107 -0.154523
    1.436181 0.598326
    -1.580597 -0.632704
    0.198447 0.011661
    -0.151489 0.044162
    1.193648 0.351535
    -0.269436 -0.063876
    0.293281 0.202289
    0.724435 0.316714
    1.883516 0.727065
    -0.808620 -0.133052
    -0.780168 -0.317335
    0.989360 0.444542
    -1.349673 -0.190884
    1.349673 0.183713
    0.034878 0.029251
    -1.193648 -0.450258
    0.174937 -0.066317
    -0.389837 -0.320758
    -0.926981 -0.301420
    -0.643417 -0.302214
    0.269436 0.152962
    1.122234 0.304839
    0.011625 0.045856
    0.341305 0.087903
    -1.482131 -0.375831
    1.689987 0.650506
    -1.054308 -0.451755
    1.308752 0.431875
    0.463780 0.267500
    -1.392093 -0.411877
    -0.957872 -0.491239
    1.230856 0.443432
    -0.724435 -0.238623
    -0.222028 -0.222249
    1.054308 0.264453
    -1.633728 -0.667049
    -1.230856 -0.619387
    -1.021489 -0.385956
    -1.269188 -0.181912
    0.590889 0.163155
    -0.058143 -0.093561
    0.697107 0.337387
    -0.463780 -0.055559
    0.539384 0.241174
    -1.157469 -0.536780
    -0.128093 -0.207855
    -0.488780 -0.120163
    0.222028 -0.083450
    -0.198447 -0.071407
    0.365502 0.138340
    -1.122234 -0.551703
    0.670107 0.301073
    -0.341305 -0.267167
    0.513977 0.220003
    -1.749894 -0.569933
    1.580597 0.527434
    -0.989360 -0.375296
    -1.959264 -0.567434
    0.081429 -0.187195
    0.151489 0.054287
    1.392093 0.568283
    -1.308752 -0.512905
    0.752115 0.446732
    1.021489 0.126670
    -0.752115 -0.301432
    0.896645 0.309920
    0.617016 0.303761
    0.565017 0.236582
    -0.293281 0.055310
    -0.513977 -0.154205
    1.749894 0.687061
    -0.034878 -0.054530
    0.104743 -0.164889
    0.488780 0.494324
    -1.087869 -0.475759
    -0.104743 0.098425
    -1.883516 -0.656464
    -1.814115 -0.396326
    1.269188 0.205889
    -1.530177 -0.584922
    1.959264 0.643878
    -0.590889 -0.528414
    -1.689987 -0.605731
    0.780168 0.181237
    0.643417 0.344046
    -0.670107 -0.311914
    1.814115 0.736194
    -0.174937 -0.033207
    1.633728 0.520442
    0.414321 -0.022694
    0.128093 0.003861
    -0.837497 -0.542978
    -0.539384 -0.153282
    -0.565017 -0.249435
    -0.866828 -0.393330
    -0.245687 -0.456192
    -0.011625 -0.026500
    -0.081429 -0.023749
    -0.438965 -0.050361
    0.957872 0.558749
    0.245687 0.159537
    0.207316 -0.055029
    0.754142 -0.394283
    1.233075 -0.734468
    0.036446 0.029682
    -1.348184 1.027477
    -0.256630 0.249472
    1.058880 -0.795511
    0.182766 0.159944
    0.615934 -0.952284
    0.432437 -0.746244
    0.281412 -0.009201
    -1.388795 1.081753
    -0.381586 -0.015845
    1.964307 -1.176499
    0.158276 -0.325938
    1.718267 -1.101447
    -0.870632 0.689875
    0.782709 -0.621192
    1.833677 -1.138123
    0.931345 -0.748548
    0.085087 -0.071796
    -0.615934 0.461315
    1.270417 -0.921708
    -0.811631 0.687817
    -0.589095 0.146282
    1.518947 -1.043303
    -0.133839 -0.021435
    0.306289 -0.068106
    -0.231934 0.383139
    -1.614208 0.659439
    0.356367 -0.085011
    -1.964307 1.466614
    1.196653 -0.428925
    -0.900761 0.613996
    1.614208 -1.300305
    0.811631 -0.627984
    -0.085087 -0.290047
    -0.356367 0.408809
    0.509907 -0.418023
    -0.670365 0.374916
    1.430703 -0.801345
    0.133839 -0.136535
    -0.306289 0.419538
    0.458090 -0.173780
    0.900761 -0.671188
    0.697989 -0.826731
    1.665001 -1.351004
    1.026145 -1.001546
    -1.026145 0.309634
    -0.158276 -0.269663
    0.406940 -0.032635
    -1.774351 1.109056
    -1.161082 1.017972
    -0.994003 0.642679
    -0.483910 0.527962
    -0.931345 0.600531
    1.388795 -0.838808
    0.012147 -0.074416
    0.109445 -0.192316
    -0.509907 0.463254
    -1.092249 0.644391
    1.161082 -0.314113
    -0.754142 0.727556
    -1.430703 0.971177
    -0.060757 -0.051209
    -0.182766 0.034218
    -0.432437 0.089256
    1.774351 -1.270626
    -1.896772 1.353576
    -1.474038 0.488741
    0.840931 -0.681575
    -0.562487 0.161637
    -1.058880 0.540784
    1.474038 -0.973699
    -0.697989 0.267387
    -0.036446 -0.083965
    0.256630 -0.457192
    -1.565603 0.925934
    -0.725909 0.443200
    -0.782709 0.674918
    0.562487 0.142057
    0.994003 -0.785518
    -1.718267 1.177377
    -1.126300 0.923393
    1.565603 -1.125271
    -1.518947 0.803877
    -0.536095 0.297529
    -0.406940 0.372309
    0.870632 -0.752876
    -0.331271 0.117802
    -0.458090 0.278735
    -0.012147 0.098037
    0.962415 -0.658874
    0.231934 0.259408
    -0.207316 0.363333
    1.092249 -0.610397
    -1.270417 0.872393
    0.060757 0.007911
    0.381586 -0.513768
    -1.196653 0.903093
    1.126300 -0.849355
    -1.833677 1.129293
    1.308758 -0.989037
    1.896772 -1.215831
    0.725909 -0.787177
    -1.665001 1.176331
    0.670365 -0.510040
    -1.233075 0.942632
    0.536095 -0.469315
    1.348184 -1.051312
    -0.840931 0.933665
    0.483910 -0.404353
    0.589095 -0.626296
    -0.109445 0.081802
    0.331271 -0.250526
    0.643019 -0.702560
    -1.308758 0.818560
    -0.281412 0.320231
    -0.962415 0.263095
    -0.643019 0.461674
    """
    X: Matrix = [
        [float(parts[0]), float(parts[1])]
        for parts in (line.split() for line in raw_data.strip().splitlines())
    ]

    mean = mean_vector(X)
    X_centered = center_matrix(X, mean)

    gram = gram_matrix(X_centered)
    eigenvalues, eigenvectors = eigen_decomposition_2x2(gram)

    singular_values = [math.sqrt(max(value, 0.0)) for value in eigenvalues]

    left_vectors: List[Vector] = []
    for singular_value, right_vector in zip(singular_values, eigenvectors):
        projected = matrix_vector_product(X_centered, right_vector)
        if singular_value > 1e-12:
            scaled = [value / singular_value for value in projected]
        else:
            scaled = [0.0 for value in projected]
        left_vectors.append(scaled)

    U = [[left_vectors[j][i] for j in range(len(left_vectors))] for i in range(len(X_centered))]
    Sigma = make_diagonal_matrix(singular_values)
    Vt = [vector[:] for vector in eigenvectors]

    reconstruction = matrix_multiply(matrix_multiply(U, Sigma), Vt)

    print("原始数据矩阵 X:\n" + format_matrix(X))
    print("\n均值向量:\n[" + ", ".join(f"{value: .6f}" for value in mean) + "]")
    print("\n均值中心化后的矩阵 X_centered:\n" + format_matrix(X_centered))
    print("\nSVD 分解结果:")
    print("U =\n" + format_matrix(U))
    print("Σ =\n" + format_matrix(Sigma))
    print("V^T =\n" + format_matrix(Vt))
    print("\n验证 U * Σ * V^T 是否还原 X_centered:\n" + format_matrix(reconstruction))

    # Compute reconstruction error for validation.
    max_error = max(
        abs(reconstruction[i][j] - X_centered[i][j])
        for i in range(len(X_centered))
        for j in range(len(X_centered[0]))
    )
    print(f"\n最大还原误差: {max_error:.6e}")

    principal_vectors = [
        (
            [eigenvectors[i][0] * singular_values[i], eigenvectors[i][1] * singular_values[i]],
            f"PC{i + 1}",
            color,
        )
        for i, color in enumerate(["#1f77b4", "#ff7f0e"])
    ]

    output_path = Path("pca_principal_axes.svg")
    create_svg(X, mean, principal_vectors, output_path)
    print(f"\n可视化结果已保存到 {output_path}")


if __name__ == "__main__":
    main()
