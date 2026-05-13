options(stringsAsFactors = FALSE)

root <- normalizePath(file.path(getwd()), winslash = "/", mustWork = TRUE)
data_dir <- file.path(root, "analysis", "data")
fig_dir <- file.path(root, "analysis", "figures")
tab_dir <- file.path(root, "analysis", "tables")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(tab_dir, recursive = TRUE, showWarnings = FALSE)

history <- read.csv(file.path(data_dir, "training_history.csv"))
scores <- read.csv(file.path(data_dir, "reconstruction_scores.csv"))
splits <- read.csv(file.path(data_dir, "split_manifest.csv"))

scores$video <- factor(scores$video, levels = c("lasercmt3.5x", "lasercmt3.5y", "lasercmt3.5y-1"))
scores$is_anomaly <- as.integer(scores$is_anomaly)

summarise_numeric <- function(x) {
  c(
    n = length(x),
    mean = mean(x),
    sd = sd(x),
    median = median(x),
    q1 = unname(quantile(x, 0.25)),
    q3 = unname(quantile(x, 0.75)),
    min = min(x),
    max = max(x)
  )
}

score_summary <- do.call(
  rbind,
  lapply(split(scores$reconstruction_l1, scores$video), summarise_numeric)
)
score_summary <- data.frame(video = rownames(score_summary), score_summary, row.names = NULL)

overall <- data.frame(
  metric = c(
    "test_n",
    "threshold_p95",
    "overall_mean",
    "overall_sd",
    "overall_median",
    "overall_min",
    "overall_max",
    "anomaly_count",
    "train_final_l1",
    "val_final_l1",
    "val_relative_reduction"
  ),
  value = c(
    nrow(scores),
    unique(scores$threshold)[1],
    mean(scores$reconstruction_l1),
    sd(scores$reconstruction_l1),
    median(scores$reconstruction_l1),
    min(scores$reconstruction_l1),
    max(scores$reconstruction_l1),
    sum(scores$is_anomaly),
    tail(history$train_l1, 1),
    tail(history$val_l1, 1),
    (history$val_l1[1] - tail(history$val_l1, 1)) / history$val_l1[1]
  )
)

write.csv(score_summary, file.path(tab_dir, "score_summary_by_video.csv"), row.names = FALSE)
write.csv(overall, file.path(tab_dir, "overall_metrics.csv"), row.names = FALSE)

kw <- kruskal.test(reconstruction_l1 ~ video, data = scores)
pairwise <- pairwise.wilcox.test(
  scores$reconstruction_l1,
  scores$video,
  p.adjust.method = "BH",
  exact = FALSE
)

sink(file.path(tab_dir, "statistical_tests.txt"))
cat("Kruskal-Wallis test for reconstruction error by video group\n")
print(kw)
cat("\nPairwise Wilcoxon rank-sum tests with Benjamini-Hochberg correction\n")
print(pairwise)
cat("\nTraining final metrics\n")
print(tail(history, 1))
sink()

top <- scores[order(scores$reconstruction_l1, decreasing = TRUE), ]
write.csv(head(top, 10), file.path(tab_dir, "top_anomaly_candidates.csv"), row.names = FALSE)

nature_cols <- c("#0072B2", "#D55E00", "#009E73")

png(file.path(fig_dir, "fig1_training_curve.png"), width = 1800, height = 1200, res = 300)
par(mar = c(4.5, 5, 2, 1), family = "sans", lwd = 1.5)
plot(history$epoch, history$train_l1, type = "o", pch = 16, col = nature_cols[1],
     xlab = "Epoch", ylab = "L1 reconstruction loss", ylim = range(c(history$train_l1, history$val_l1)),
     cex.lab = 1.1, cex.axis = 0.95)
lines(history$epoch, history$val_l1, type = "o", pch = 17, col = nature_cols[2])
legend("topright", legend = c("Training", "Validation"), col = nature_cols[1:2], pch = c(16, 17),
       lty = 1, bty = "n", cex = 0.9)
box(bty = "l")
dev.off()

png(file.path(fig_dir, "fig2_score_distribution_by_video.png"), width = 1800, height = 1200, res = 300)
par(mar = c(5.5, 5, 2, 1), family = "sans")
boxplot(reconstruction_l1 ~ video, data = scores, col = adjustcolor(nature_cols, 0.35),
        border = nature_cols, ylab = "Reconstruction error (L1)", xlab = "",
        outline = FALSE, cex.axis = 0.9, cex.lab = 1.1)
stripchart(reconstruction_l1 ~ video, data = scores, vertical = TRUE, method = "jitter",
           pch = 21, bg = "white", col = "gray20", add = TRUE)
abline(h = unique(scores$threshold)[1], lty = 2, col = "gray30", lwd = 1.3)
legend("topleft", legend = "95th percentile threshold", lty = 2, col = "gray30", bty = "n")
box(bty = "l")
dev.off()

png(file.path(fig_dir, "fig3_ranked_anomaly_scores.png"), width = 1800, height = 1200, res = 300)
par(mar = c(4.5, 5, 2, 1), family = "sans")
ranked <- scores[order(scores$reconstruction_l1, decreasing = TRUE), ]
bar_cols <- nature_cols[as.integer(ranked$video)]
barplot(ranked$reconstruction_l1, col = bar_cols, border = NA,
        ylab = "Reconstruction error (L1)", xlab = "Ranked test frames",
        cex.lab = 1.1, cex.axis = 0.95)
abline(h = unique(scores$threshold)[1], lty = 2, col = "gray20", lwd = 1.3)
legend("topright", legend = levels(scores$video), fill = nature_cols, bty = "n", cex = 0.85)
box(bty = "l")
dev.off()

png(file.path(fig_dir, "fig4_train_test_composition.png"), width = 1800, height = 1200, res = 300)
par(mar = c(5, 5, 2, 1), family = "sans")
split_table <- table(splits$split, splits$video)
barplot(t(split_table), beside = TRUE, col = nature_cols, border = NA,
        ylab = "Number of frames", xlab = "Dataset split", cex.lab = 1.1, cex.axis = 0.95)
legend("topright", legend = colnames(split_table), fill = nature_cols, bty = "n", cex = 0.85)
box(bty = "l")
dev.off()

cat("Analysis complete.\n")
cat("Figures:", fig_dir, "\n")
cat("Tables:", tab_dir, "\n")
