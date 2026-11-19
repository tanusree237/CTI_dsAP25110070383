import java.awt.BorderLayout;
import java.awt.Button;
import java.awt.Frame;
import java.awt.Panel;
import java.awt.TextArea;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.time.LocalDate;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public record CtiRecord(String id, String provider, LocalDate published, String text) {
    public enum Label { DUPLICATE, RELATED, UNRELATED }

    public static final class Normalizer {
        private static final Pattern IPV4 = Pattern.compile("\\b(?:\\d{1,3}\\.){3}\\d{1,3}\\b");
        // Any alphabetic TLD, not just a fixed list
        private static final Pattern DOMAIN = Pattern.compile(
                "\\b[a-z0-9-]+(?:\\.[a-z0-9-]+)*\\.[a-z]{2,}\\b");
        private static final Pattern HASH = Pattern.compile("\\b(?:[a-f0-9]{32}|[a-f0-9]{40}|[a-f0-9]{64})\\b");
        private static final Pattern CVE = Pattern.compile("\\bcve-\\d{4}-\\d{4,}\\b");

        private static final Map<String, String> ALIASES = Map.of(
                "apt28", "apt28",
                "fancy bear", "apt28",
                "sofacy", "apt28",
                "lazarus group", "lazarus group",
                "hidden cobra", "lazarus group",
                "emotet", "emotet",
                "cobalt strike", "cobalt strike",
                "lockbit", "lockbit");

        // Compiled once instead of on every call
        private static final Map<String, Pattern> ALIAS_PATTERNS = new HashMap<>();
        static {
            for (String alias : ALIASES.keySet()) {
                ALIAS_PATTERNS.put(alias,
                        Pattern.compile("\\b" + Pattern.quote(alias) + "\\b"));
            }
        }

        private Normalizer() {}

        public static String normalize(String raw) {
            return raw == null ? "" : raw.toLowerCase(Locale.ROOT)
                    .replace("[.]", ".")
                    .replace("hxxp", "http")
                    .replaceAll("\\s+", " ")
                    .trim();
        }

        public static Set<String> indicators(String raw) {
            String s = normalize(raw);
            Set<String> out = new TreeSet<>();
            collect(out, "ip:", IPV4, s);
            collect(out, "domain:", DOMAIN, s);
            collect(out, "hash:", HASH, s);
            collect(out, "cve:", CVE, s);
            return out;
        }

        public static Set<String> entities(String raw) {
            String s = normalize(raw);
            Set<String> out = new TreeSet<>();
            for (Map.Entry<String, String> e : ALIASES.entrySet()) {
                if (ALIAS_PATTERNS.get(e.getKey()).matcher(s).find()) {
                    out.add(e.getValue());
                }
            }
            return out;
        }

        public static List<String> tokens(String raw) {
            String s = normalize(raw).replaceAll("[^a-z0-9.\\- ]", " ");
            List<String> out = new ArrayList<>();
            for (String t : s.split("\\s+")) {
                if (t.length() > 1) {
                    out.add(t);
                }
            }
            return out;
        }

        private static void collect(Set<String> out, String prefix, Pattern p, String s) {
            Matcher m = p.matcher(s);
            while (m.find()) {
                out.add(prefix + m.group());
            }
        }
    }

    public static final class PairScorer {
        public record Features(double indicatorJaccard, double tfidfCosine,
                               double entityJaccard, boolean entityConflict) {}

        private final Map<String, Double> idf = new HashMap<>();

        public PairScorer(Collection<CtiRecord> corpus) {
            Map<String, Integer> df = new HashMap<>();
            for (CtiRecord r : corpus) {
                for (String t : new HashSet<>(Normalizer.tokens(r.text()))) {
                    df.merge(t, 1, Integer::sum);
                }
            }
            int n = corpus.size();
            df.forEach((t, c) -> idf.put(t, Math.log((1.0 + n) / (1.0 + c)) + 1.0));
        }

        public Features features(CtiRecord a, CtiRecord b) {
            Set<String> ea = Normalizer.entities(a.text());
            Set<String> eb = Normalizer.entities(b.text());
            boolean conflict = !ea.isEmpty() && !eb.isEmpty() && Collections.disjoint(ea, eb);
            return new Features(
                    jaccard(Normalizer.indicators(a.text()), Normalizer.indicators(b.text())),
                    cosine(tfidf(a.text()), tfidf(b.text())),
                    jaccard(ea, eb),
                    conflict);
        }

        // Single source of truth for the weighted score
        public double score(Features f) {
            return 0.4 * f.indicatorJaccard() + 0.4 * f.tfidfCosine() + 0.2 * f.entityJaccard();
        }

        public Label classify(Features f) {
            double score = score(f);

            if (f.entityConflict()) {
                return score >= 0.2 ? Label.RELATED : Label.UNRELATED;
            }
            if (score >= 0.6 && (f.tfidfCosine() >= 0.5 || f.indicatorJaccard() >= 0.6)) {
                return Label.DUPLICATE;
            }
            return score >= 0.3 ? Label.RELATED : Label.UNRELATED;
        }

        private Map<String, Double> tfidf(String text) {
            Map<String, Double> tf = new HashMap<>();
            for (String t : Normalizer.tokens(text)) {
                tf.merge(t, 1.0, Double::sum);
            }
            Map<String, Double> vec = new HashMap<>();
            for (Map.Entry<String, Double> e : tf.entrySet()) {
                vec.put(e.getKey(), e.getValue() * idf.getOrDefault(e.getKey(), 1.0));
            }
            return vec;
        }

        private static double cosine(Map<String, Double> a, Map<String, Double> b) {
            double dot = 0, na = 0, nb = 0;
            for (Map.Entry<String, Double> e : a.entrySet()) {
                na += e.getValue() * e.getValue();
                Double o = b.get(e.getKey());
                if (o != null) dot += e.getValue() * o;
            }
            for (double v : b.values()) nb += v * v;
            return (na == 0 || nb == 0) ? 0.0 : dot / (Math.sqrt(na) * Math.sqrt(nb));
        }

        private static double jaccard(Set<String> a, Set<String> b) {
            if (a.isEmpty() && b.isEmpty()) return 0.0;
            Set<String> inter = new HashSet<>(a);
            inter.retainAll(b);
            Set<String> union = new HashSet<>(a);
            union.addAll(b);
            return union.isEmpty() ? 0.0 : (double) inter.size() / union.size();
        }
    }

    public static final class Deduplicator {
        public record Cluster(List<CtiRecord> members, Set<String> indicators,
                             Set<String> entities, List<String> conflicts) {}

        public static List<Cluster> run(List<CtiRecord> records, PairScorer scorer) {
            int n = records.size();
            int[] parent = new int[n];
            for (int i = 0; i < n; i++) parent[i] = i;

            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    PairScorer.Features f = scorer.features(records.get(i), records.get(j));
                    if (scorer.classify(f) == Label.DUPLICATE) {
                        union(parent, i, j);
                    }
                }
            }

            Map<Integer, List<CtiRecord>> groups = new LinkedHashMap<>();
            for (int i = 0; i < n; i++) {
                groups.computeIfAbsent(find(parent, i), k -> new ArrayList<>()).add(records.get(i));
            }

            List<Cluster> out = new ArrayList<>();
            for (List<CtiRecord> g : groups.values()) {
                out.add(merge(g));
            }
            return out;
        }

        private static Cluster merge(List<CtiRecord> members) {
            Set<String> indicators = new TreeSet<>();
            Set<String> entities = new TreeSet<>();
            Map<String, String> attribution = new TreeMap<>();
            Set<Set<String>> distinctAttributions = new HashSet<>();

            for (CtiRecord r : members) {
                indicators.addAll(Normalizer.indicators(r.text()));
                Set<String> e = Normalizer.entities(r.text());
                entities.addAll(e);
                if (!e.isEmpty()) {
                    distinctAttributions.add(new TreeSet<>(e));
                }
                for (String x : e) {
                    attribution.merge(x, r.provider(), (a, b) -> a + ", " + b);
                }
            }

            List<String> conflicts = new ArrayList<>();
            if (distinctAttributions.size() > 1) {
                conflicts.add("members disagree on attribution: " + attribution);
            }
            return new Cluster(members, indicators, entities, conflicts);
        }

        private static int find(int[] p, int x) {
            while (p[x] != x) {
                p[x] = p[p[x]];
                x = p[x];
            }
            return x;
        }

        private static void union(int[] p, int a, int b) {
            p[find(p, a)] = find(p, b);
        }
    }

    public static final class Evaluator {
        public static void runDemo() {
            CtiRecord a = new CtiRecord(
                    "CTI-1001",
                    "Acme SOC",
                    LocalDate.of(2025, 1, 14),
                    "APT28 used phishing pages to steal finance credentials and deploy Emotet. Domain login.acme-secure-check.com and cve-2024-1234."
            );
            CtiRecord b = new CtiRecord(
                    "CTI-1002",
                    "Threat Intel",
                    LocalDate.of(2025, 1, 15),
                    "Sofacy phishing against employees targeting banking credentials and Emotet distribution. login.acme-secure-check.com"
            );
            CtiRecord c = new CtiRecord(
                    "CTI-2001",
                    "External Feed",
                    LocalDate.of(2025, 2, 10),
                    "Ransomware with lockbit and remote desktop access, unrelated to the APT28 activity."
            );

            List<CtiRecord> records = List.of(a, b, c);
            PairScorer scorer = new PairScorer(records);
            List<Deduplicator.Cluster> clusters = Deduplicator.run(records, scorer);

            StringBuilder output = new StringBuilder();
            output.append("CTI Record Deduplicator\n")
                    .append("=======================\n\n");
            for (CtiRecord record : records) {
                output.append(record.id()).append(" | ").append(record.provider())
                        .append(" | ").append(record.published()).append("\n")
                        .append(record.text()).append("\n\n");
            }

            appendPair(output, "A vs B", scorer, a, b);
            appendPair(output, "A vs C", scorer, a, c);

            output.append("\nClusters found: ").append(clusters.size()).append("\n");

            int clusterNumber = 1;
            for (Deduplicator.Cluster cluster : clusters) {
                output.append("\nCluster ").append(clusterNumber++).append(": ");
                output.append(cluster.members().stream()
                        .map(CtiRecord::id)
                        .toList()).append("\n");
                output.append("Indicators: ").append(cluster.indicators()).append("\n");
                output.append("Entities: ").append(cluster.entities()).append("\n");
                if (!cluster.conflicts().isEmpty()) {
                    output.append("Conflicts: ").append(cluster.conflicts()).append("\n");
                }
            }

            showAwtWindow(output.toString());
        }

        // Scores each pair once and reuses the result for the label and the printed score
        private static void appendPair(StringBuilder out, String name,
                                       PairScorer scorer, CtiRecord x, CtiRecord y) {
            PairScorer.Features f = scorer.features(x, y);
            out.append(name).append(" => ").append(scorer.classify(f))
                    .append(" score=").append(String.format(Locale.US, "%.3f", scorer.score(f)))
                    .append("\n");
        }

        private static void showAwtWindow(String output) {
            Frame frame = new Frame("CTI Record Deduplicator");
            frame.setLayout(new BorderLayout(10, 10));
            frame.setSize(720, 520);

            Panel header = new Panel();
            header.add(new java.awt.Label("CTI Record Analysis"));
            frame.add(header, BorderLayout.NORTH);

            TextArea textArea = new TextArea(output, 24, 80, TextArea.SCROLLBARS_BOTH);
            textArea.setEditable(false);
            frame.add(textArea, BorderLayout.CENTER);

            Panel footer = new Panel();
            Button closeButton = new Button("Close");
            closeButton.addActionListener(event -> frame.dispose());
            footer.add(closeButton);
            frame.add(footer, BorderLayout.SOUTH);

            frame.addWindowListener(new WindowAdapter() {
                @Override
                public void windowClosing(WindowEvent event) {
                    frame.dispose();
                }
            });

            frame.setLocationByPlatform(true);
            frame.setVisible(true);
        }
    }

    public static void main(String[] args) {
        java.awt.EventQueue.invokeLater(Evaluator::runDemo);
    }
}