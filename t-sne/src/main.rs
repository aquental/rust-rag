mod data;
mod embeddings;

use bhtsne;
use embeddings::SentenceEmbedder;
use plotters::prelude::full_palette::PURPLE;
use plotters::prelude::*;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (sentences, categories) = data::get_sentences_and_categories();

    let sentence_refs: Vec<&str> = sentences.iter().map(String::as_str).collect();
    let result = compute_tsne_embeddings(&sentence_refs).await?;

    println!("\n2D Embeddings:");
    for (i, coords) in result.chunks(2).enumerate() {
        println!(
            "Sentence {}: [{:.4}, {:.4}] ({})",
            i + 1,
            coords[0],
            coords[1],
            categories[i]
        );
    }

    plot_tsne_embedding(&result, &sentences, &categories)?;
    Ok(())
}

fn get_color_and_shape_maps() -> (
    HashMap<&'static str, RGBColor>,
    HashMap<&'static str, ShapeStyle>,
) {
    let color_map = HashMap::from([
        ("NLP", RED),
        ("ML", BLUE),
        ("Food", GREEN),
        ("Weather", PURPLE),
    ]);

    let shape_map = HashMap::from([
        ("NLP", RED.filled()),
        ("ML", BLUE.filled()),
        ("Food", GREEN.filled()),
        ("Weather", PURPLE.filled()),
    ]);

    (color_map, shape_map)
}

async fn compute_tsne_embeddings(
    sentences: &[&str],
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let embedder = SentenceEmbedder::new().await?;
    let embeddings = embedder.embed_texts(sentences)?;
    let samples: Vec<&[f32]> = embeddings.iter().map(Vec::as_slice).collect();

    let mut tsne = bhtsne::tSNE::new(&samples);
    tsne.embedding_dim(2)
        .perplexity(10.0)
        .epochs(3000)
        .barnes_hut(0.5, |a, b| {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt()
        });

    Ok(tsne.embedding())
}

fn plot_tsne_embedding(
    points: &[f32],
    labels: &[String],
    categories: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    let (color_map, _) = get_color_and_shape_maps();

    let root = BitMapBackend::new("tsne_plot.png", (1000, 1000)).into_drawing_area();
    root.fill(&WHITE)?;

    let xs: Vec<f32> = points
        .iter()
        .copied()
        .enumerate()
        .filter(|(i, _)| i % 2 == 0)
        .map(|(_, x)| x)
        .collect();
    let ys: Vec<f32> = points
        .iter()
        .copied()
        .enumerate()
        .filter(|(i, _)| i % 2 == 1)
        .map(|(_, y)| y)
        .collect();

    let padding = 20.0;
    let x_min = xs.iter().cloned().fold(f32::INFINITY, f32::min) - padding;
    let x_max = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max) + padding;
    let y_min = ys.iter().cloned().fold(f32::INFINITY, f32::min) - padding;
    let y_max = ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max) + padding;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "t-SNE Visualization of Sentence Embeddings",
            ("sans-serif", 14),
        )
        .margin(25)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart.configure_mesh().draw()?;

    for ((x, y), (label, category)) in xs
        .iter()
        .zip(ys.iter())
        .zip(labels.iter().zip(categories.iter()))
    {
        let color = color_map.get(category.as_str()).unwrap_or(&BLACK);

        match category.as_str() {
            "NLP" => {
                chart.draw_series(std::iter::once(Circle::new((*x, *y), 5, color.filled())))?;
            }
            "ML" => {
                chart.draw_series(std::iter::once(Rectangle::new(
                    [(*x - 3.0, *y - 3.0), (*x + 3.0, *y + 3.0)],
                    color.filled(),
                )))?;
            }
            "Food" => {
                chart.draw_series(std::iter::once(Polygon::new(
                    vec![(*x, *y - 4.0), (*x - 4.0, *y + 4.0), (*x + 4.0, *y + 4.0)],
                    color.filled(),
                )))?;
            }
            "Weather" => {
                chart.draw_series(std::iter::once(Polygon::new(
                    vec![
                        (*x, *y - 4.0),
                        (*x + 4.0, *y),
                        (*x, *y + 4.0),
                        (*x - 4.0, *y),
                    ],
                    color.filled(),
                )))?;
            }
            _ => {
                chart.draw_series(std::iter::once(Circle::new((*x, *y), 5, BLACK.filled())))?;
            }
        }

        chart.draw_series(std::iter::once(Text::new(
            format!("{}...", label.chars().take(20).collect::<String>()),
            (*x + 3.0, *y - 7.5),
            ("sans-serif", 14),
        )))?;
    }

    root.present()?;
    println!("Saved plot to tsne_plot.png");
    Ok(())
}
