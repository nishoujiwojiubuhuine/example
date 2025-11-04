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
    # Construct a small 2D dataset (rows: samples, columns: features).
    X: Matrix = [
        [2.5, 2.4],
        [0.5, 0.7],
        [2.2, 2.9],
        [1.9, 2.2],
        [3.1, 3.0],
        [2.3, 2.7],
        [2.0, 1.6],
        [1.0, 1.1],
        [1.5, 1.6],
        [1.1, 0.9],
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
