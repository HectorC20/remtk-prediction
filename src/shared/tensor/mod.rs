use half::f16;

/// Normalización L2 unitaria de un vector unidimensional en el espacio de Hilbert.
pub fn l2_normalize(vec: &mut [f32]) {
    let norm: f32 = vec.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-12);
    for x in vec.iter_mut() {
        *x /= norm;
    }
}

/// Cálculo de similitud coseno entre dos vectores ya normalizados L2.
pub fn cosine_similarity_normalized(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Convierte un slice continuo de flotantes de media precisión (half::f16) a FP32.
pub fn f16_slice_to_f32(src: &[f16], dst: &mut [f32]) {
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = s.to_f32();
    }
}

/// Convierte un slice de FP32 a media precisión (half::f16) para almacenamiento en caché.
pub fn f32_slice_to_f16(src: &[f32], dst: &mut Vec<f16>) {
    dst.clear();
    dst.reserve(src.len());
    for &s in src {
        dst.push(f16::from_f32(s));
    }
}
