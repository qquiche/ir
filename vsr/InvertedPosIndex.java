package ir.vsr;

import java.io.File;
import java.util.*;
import ir.utilities.*;
import ir.classifiers.*;

/**
 * Standalone proximity-enhanced inverted index.
 *
 * Usage (same flags as baseline):
 *   javac ir/vsr/*.java
 *   java ir.vsr.InvertedPosIndex -html <path-to-dataset>
 *
 * Prints: Score: <final> (Vector: <cos>; Proximity: <prox>)
 */
public class InvertedPosIndex {

  /** ========= Config (no extra CLI flags) ========= */
  public static final int MAX_RETRIEVALS = 10;  // same paging as baseline

  private static final double LAMBDA        = 0.25; // blend for avg pairwise distance
  private static final double ORDER_PENALTY = 1.8;  // penalty for reverse order
  private static final double BIG_S         = 1_000_000.0; // penalty when terms missing

  /** ========= Core index state (mirrors baseline) ========= */
  public Map<String, TokenInfo> tokenHash = new HashMap<>();
  public List<DocumentReference> docRefs   = new ArrayList<>();

  public File  dirFile;
  public short docType = DocumentIterator.TYPE_TEXT;
  public boolean stem = false;
  public boolean feedback = false;

  /** ========= Positional index ========= */
  public static class PosPosting {
    final DocumentReference docRef;
    final int tf;
    final int[] positions; // strictly increasing in filtered-token space
    PosPosting(DocumentReference d, int tf, int[] p){ this.docRef=d; this.tf=tf; this.positions=p; }
  }
  public static class PosTokenInfo {
    double idf = 0.0;
    final List<PosPosting> occList = new ArrayList<>();
  }
  public Map<String, PosTokenInfo> posTokenHash = new HashMap<>();

  /** ========= Constructors ========= */
  public InvertedPosIndex(File dirFile, short docType, boolean stem, boolean feedback) {
    this.dirFile = dirFile;
    this.docType = docType;
    this.stem = stem;
    this.feedback = feedback;
    indexDocuments();
  }

  public InvertedPosIndex(List<Example> examples) {
    indexDocuments(examples);
  }

  /** ========= Build: index the directory (mirrors baseline flow) ========= */
  protected void indexDocuments() {
    if (!tokenHash.isEmpty() || !docRefs.isEmpty()) {
      throw new IllegalStateException("Cannot indexDocuments more than once.");
    }
    DocumentIterator docIter = new DocumentIterator(dirFile, docType, stem);
    System.out.println("Indexing documents in " + dirFile);
    while (docIter.hasMoreDocuments()) {
      FileDocument doc = docIter.nextDocument();
      System.out.print(doc.file.getName() + ",");
      HashMapVector vector = doc.hashMapVector();
      indexDocument(doc, vector);
    }
    computeIDFandDocumentLengths();
    System.out.println("\nIndexed " + docRefs.size() + " documents with " + size() + " unique terms.");
  }

  /** Index a list of Examples (kept for API parity). */
  protected void indexDocuments(List<Example> examples) {
    if (!tokenHash.isEmpty() || !docRefs.isEmpty()) {
      throw new IllegalStateException("Cannot indexDocuments more than once.");
    }
    for (Example example : examples) {
      FileDocument doc = example.getDocument();
      HashMapVector vector = example.getHashMapVector();
      indexDocument(doc, vector);
    }
    computeIDFandDocumentLengths();
    System.out.println("Indexed " + docRefs.size() + " documents with " + size() + " unique terms.");
  }

  /** Index a single document: baseline postings + positional postings. */
  protected void indexDocument(FileDocument doc, HashMapVector vector) {
    // ---- Baseline: add docRef & BoW postings ----
    DocumentReference docRef = new DocumentReference(doc);
    docRefs.add(docRef);

    for (Map.Entry<String, Weight> entry : vector.entrySet()) {
      String token = entry.getKey();
      int count = (int) entry.getValue().getValue();
      TokenInfo tokenInfo = tokenHash.get(token);
      if (tokenInfo == null) {
        tokenInfo = new TokenInfo();
        tokenHash.put(token, tokenInfo);
      }
      tokenInfo.occList.add(new TokenOccurrence(docRef, count));
    }

    // ---- Positional: re-tokenize with identical filtering and record positions ----
    Document posDoc = (this.docType == DocumentIterator.TYPE_HTML)
    ? new RawHTMLFileDocument(doc.file)   // implement like RawTextFileDocument
    : new RawTextFileDocument(doc.file);


    Map<String, List<Integer>> term2pos = new HashMap<>();
    int filteredPos = 0;
    while (posDoc.hasMoreTokens()) {
      String tok = posDoc.nextToken(); // filtered (stopwords removed; optionally stemmed)
      if (tok == null) break;
      term2pos.computeIfAbsent(tok, k -> new ArrayList<>()).add(filteredPos++);
    }

    for (Map.Entry<String, List<Integer>> e : term2pos.entrySet()) {
      int[] pos = e.getValue().stream().mapToInt(Integer::intValue).toArray();
      Arrays.sort(pos);
      PosTokenInfo pti = posTokenHash.computeIfAbsent(e.getKey(), k -> new PosTokenInfo());
      pti.occList.add(new PosPosting(docRef, pos.length, pos));
    }
  }

  /** Compute IDF and doc lengths (same math as baseline) and mirror IDF into positional map. */
  protected void computeIDFandDocumentLengths() {
    double N = docRefs.size();
    Iterator<Map.Entry<String, TokenInfo>> it = tokenHash.entrySet().iterator();
    while (it.hasNext()) {
      Map.Entry<String, TokenInfo> entry = it.next();
      TokenInfo tokenInfo = entry.getValue();
      double numDocRefs = tokenInfo.occList.size();
      double idf = Math.log(N / numDocRefs);
      if (idf == 0.0) {
        it.remove();
      } else {
        tokenInfo.idf = idf;
        for (TokenOccurrence occ : tokenInfo.occList) {
          occ.docRef.length = occ.docRef.length + Math.pow(idf * occ.count, 2);
        }
      }
    }
    for (DocumentReference docRef : docRefs) {
      docRef.length = Math.sqrt(docRef.length);
    }
    // Mirror IDF to positional map
    for (Map.Entry<String, TokenInfo> e : tokenHash.entrySet()) {
      PosTokenInfo pti = posTokenHash.get(e.getKey());
      if (pti != null) pti.idf = e.getValue().idf;
    }
  }

  /** ========= Retrieval API (string/doc/vector), with proximity-aware final scoring ========= */

  public Retrieval[] retrieve(String input) {
    // Build two docs to preserve order and get vector on same pipeline
    TextStringDocument orderDoc = new TextStringDocument(input, stem);
    List<String> qOrder = extractOrderedUniqueTokens(orderDoc);

    TextStringDocument vectorDoc = new TextStringDocument(input, stem);
    HashMapVector qv = vectorDoc.hashMapVector();

    return retrieve(qv, qOrder);
  }

  public Retrieval[] retrieve(Document doc) {
    // Fallback: we don't have raw query text here for a second pass; use vector and alpha order
    return retrieve(doc.hashMapVector());
  }

  public Retrieval[] retrieve(HashMapVector vector) {
    // Deterministic fallback order for terms (used only if no explicit order provided)
    List<String> qTerms = new ArrayList<>(vector.hashMap.keySet());
    Collections.sort(qTerms);
    return retrieve(vector, qTerms);
  }

/** Core retrieval with a known query term order. */
protected Retrieval[] retrieve(HashMapVector queryVector, List<String> qOrder) {
  // ---- Stage 1: cosine accumulation over ALL tokens (baseline math) ----
  Map<DocumentReference, DoubleValue> retrievalHash = new HashMap<>();
  double queryLength = 0.0;

  for (Map.Entry<String, Weight> entry : queryVector.entrySet()) {
    String token = entry.getKey();
    double count = entry.getValue().getValue();
    queryLength += incorporateToken(token, count, retrievalHash);
  }
  queryLength = Math.sqrt(queryLength);

  // Convert to array of cosine-scored Retrievals (normalized)
  RetrievalProx[] candidates = new RetrievalProx[retrievalHash.size()];
  int idx = 0;
  for (Map.Entry<DocumentReference, DoubleValue> e : retrievalHash.entrySet()) {
    DocumentReference d = e.getKey();
    double cosine = e.getValue().value / (queryLength * d.length);
    candidates[idx++] = new RetrievalProx(d, cosine, cosine, 0.0);
  }

  // ---- Stage 2: compute proximity distance and final score for ALL candidates ----
  for (RetrievalProx rp : candidates) {
    boolean hasAll = true;
    List<int[]> posLists = new ArrayList<>(qOrder.size());
    for (String qt : qOrder) {
      int[] arr = positionsOf(qt, rp.docRef);
      if (arr == null || arr.length == 0) { hasAll = false; break; }
      posLists.add(arr);
    }

    if (!hasAll || qOrder.size() < 2) {
    rp.prox = 0.0;
    rp.score = rp.cosine / BIG_S;
    } else {
        double proxDist = computeProximityDistance(posLists);

        if (proxDist == Double.POSITIVE_INFINITY) {
            rp.prox = 0.0;
            rp.score = rp.cosine / BIG_S;
        } else {
            rp.prox = 1.0 / proxDist;         // for reporting
            rp.score = rp.cosine / proxDist;  // final score
        }
    }

  }

  // ---- Stage 3: sort final scores (best to worst) ----
  Arrays.sort(candidates);

  return candidates;
}

/**
 * Compute the minimum span (in tokens) that covers all query terms in order.
 * Returns Double.POSITIVE_INFINITY if the document does not contain all terms in order.
 */
protected static double computeProximityDistance(List<int[]> posLists) {
    int k = posLists.size();
    if (k < 2) return Double.POSITIVE_INFINITY;

    double best = Double.POSITIVE_INFINITY;

    // For every occurrence of the first query term
    for (int start : posLists.get(0)) {
        int cur = start;
        boolean ok = true;

        // Walk through the remaining terms in query order
        for (int i = 1; i < k; i++) {
            int[] positions = posLists.get(i);

            // Binary search to find the first occurrence >= cur
            int idx = Arrays.binarySearch(positions, cur);
            int ins = (idx >= 0) ? idx : -idx - 1;

            if (ins >= positions.length) {
                ok = false; break; // can't complete chain
            }
            cur = positions[ins];
        }

        if (ok) {
            int span = cur - start + 1;
            if (span < best) best = span;
        }
    }

    return best;
}




  /**
   * Baseline token incorporation: update retrievalHash with contributions of this token.
   * Returns squared weight of the token for queryLength accumulation.
   */
  protected double incorporateToken(String token, double count,
                                    Map<DocumentReference, DoubleValue> retrievalHash) {
    TokenInfo tokenInfo = tokenHash.get(token);
    if (tokenInfo == null) return 0.0;
    double weight = tokenInfo.idf * count;
    for (TokenOccurrence occ : tokenInfo.occList) {
      DoubleValue val = retrievalHash.get(occ.docRef);
      if (val == null) {
        val = new DoubleValue(0.0);
        retrievalHash.put(occ.docRef, val);
      }
      val.value = val.value + weight * tokenInfo.idf * occ.count;
    }
    return weight * weight;
  }

  /** ========= Interactive query processing & printing (kept like baseline) ========= */

  public void processQueries() {
    System.out.println("Now able to process queries. When done, enter an empty query to exit.");
    do {
      String query = UserInput.prompt("\nEnter query:  ");
      if (query.equals("")) break;

      TextStringDocument orderDoc  = new TextStringDocument(query, stem);
      List<String> qOrder = extractOrderedUniqueTokens(orderDoc);

      TextStringDocument vectorDoc = new TextStringDocument(query, stem);
      HashMapVector queryVector = vectorDoc.hashMapVector();

      Retrieval[] retrievals = retrieve(queryVector, qOrder);
      presentRetrievals(queryVector, retrievals);
    } while (true);
  }

  public void presentRetrievals(HashMapVector queryVector, Retrieval[] retrievals) {
    if (showRetrievals(retrievals)) {
      FeedbackProx fdback = null;
      if (feedback) fdback = new FeedbackProx(queryVector, retrievals, this);
      int currentPosition = MAX_RETRIEVALS;
      int showNumber = 0;
      do {
        String command = UserInput.prompt("\n Enter command:  ");
        if (command.equals("")) break;
        if (command.equals("m")) {
          printRetrievals(retrievals, currentPosition);
          currentPosition += MAX_RETRIEVALS;
          continue;
        }
        if (command.equals("r") && feedback) {
          if (fdback.isEmpty()) {
            System.out.println("Need to first view some documents and provide feedback.");
            continue;
          }
          System.out.println("Positive docs: " + fdback.goodDocRefs +
              "\nNegative docs: " + fdback.badDocRefs);
          System.out.println("Executing New Expanded and Reweighted Query: ");
          queryVector = fdback.newQuery();
          retrievals = retrieve(queryVector); // order fallback path for feedback
          fdback.retrievals = retrievals;
          if (showRetrievals(retrievals)) continue;
          else break;
        }
        try {
          showNumber = Integer.parseInt(command);
        } catch (NumberFormatException e) {
          System.out.println("Unknown command.");
          System.out.println("Enter `m' to see more, a number to show the nth document, nothing to exit.");
          if (feedback && !fdback.isEmpty())
            System.out.println("Enter `r' to use any feedback given to `redo' with a revised query.");
          continue;
        }
        if (showNumber > 0 && showNumber <= retrievals.length) {
          System.out.println("Showing document " + showNumber + " in the " + Browser.BROWSER_NAME + " window.");
          Browser.display(retrievals[showNumber - 1].docRef.file);
          if (feedback && !fdback.haveFeedback(showNumber)) fdback.getFeedback(showNumber);
        } else {
          System.out.println("No such document number: " + showNumber);
        }
      } while (true);
    }
  }

  public boolean showRetrievals(Retrieval[] retrievals) {
    if (retrievals.length == 0) {
      System.out.println("\nNo matching documents found.");
      return false;
    } else {
      System.out.println("\nTop " + MAX_RETRIEVALS + " matching Documents from most to least relevant:");
      printRetrievals(retrievals, 0);
      System.out.println("\nEnter `m' to see more, a number to show the nth document, nothing to exit.");
      if (feedback)
        System.out.println("Enter `r' to use any relevance feedback given to `redo' with a revised query.");
      return true;
    }
  }

  public void printRetrievals(Retrieval[] retrievals, int start) {
    System.out.println("");
    if (start >= retrievals.length) {
      System.out.println("No more retrievals.");
      return;
    }
    for (int i = start; i < Math.min(retrievals.length, start + MAX_RETRIEVALS); i++) {
      Retrieval r = retrievals[i];
      if (r instanceof RetrievalProx) {
        RetrievalProx rp = (RetrievalProx) r;
        System.out.println(
            MoreString.padTo((i + 1) + ". ", 4) +
            MoreString.padTo(rp.docRef.file.getName(), 20) +
            " Score: " + MoreMath.roundTo(rp.score, 5) +
            " (Vector: " + MoreMath.roundTo(rp.cosine, 5) +
            "; Proximity: " + MoreMath.roundTo(rp.prox, 5) + ")"
        );
      } else {
        System.out.println(
            MoreString.padTo((i + 1) + ". ", 4) +
            MoreString.padTo(r.docRef.file.getName(), 20) +
            " Score: " + MoreMath.roundTo(r.score, 5)
        );
      }
    }
  }

  /** ========= Helpers ========= */

  /** Retrieval subclass carrying cosine & proximity components; score holds FINAL. */
  public static class RetrievalProx extends Retrieval {
    final double cosine; // baseline cosine
    double prox;   // 1/(1+s) reportable proximity boost
    RetrievalProx(DocumentReference d, double finalScore, double cosine, double prox) {
      super(d, finalScore);
      this.cosine = cosine;
      this.prox = prox;
    }
  }

  /** Get positions array for (term, docRef), or null if absent. */
  protected int[] positionsOf(String term, DocumentReference dref) {
    PosTokenInfo pti = posTokenHash.get(term);
    if (pti == null) return null;
    for (PosPosting p : pti.occList) if (p.docRef.equals(dref)) return p.positions;
    return null;
  }

  /** Ordered unique filtered tokens from a Document (first occurrence defines order). */
  protected static List<String> extractOrderedUniqueTokens(Document d) {
    List<String> order = new ArrayList<>();
    HashSet<String> seen = new HashSet<>();
    while (d.hasMoreTokens()) {
      String tok = d.nextToken();
      if (tok == null) break;
      if (seen.add(tok)) order.add(tok);
    }
    return order;
  }

  /** Average nearest distance over adjacent query term pairs (with reverse-order penalty). */
  protected static double avgPairwiseNearestDistance(List<int[]> posLists, double orderPenalty) {
    if (posLists.size() < 2) return 0.0;
    double sum = 0.0; int pairs = 0;
    for (int i = 0; i + 1 < posLists.size(); i++) {
      int[] A = posLists.get(i), B = posLists.get(i + 1);
      int best = Integer.MAX_VALUE;
      for (int a : A) {
        int idx = Arrays.binarySearch(B, a);
        int ins = (idx >= 0) ? idx : -idx - 1;
        if (ins < B.length) {
          int b = B[ins];
          int d = Math.abs(b - a);
          if (b < a) d = (int) Math.round(d * orderPenalty);
          if (d < best) best = d;
        }
        if (ins - 1 >= 0) {
          int b = B[ins - 1];
          int d = Math.abs(b - a);
          if (b < a) d = (int) Math.round(d * orderPenalty);
          if (d < best) best = d;
        }
      }
      if (best == Integer.MAX_VALUE) best = (int) BIG_S;
      sum += best; pairs++;
    }
    return sum / Math.max(1, pairs);
  }

  /** Returns slack s = max(0, span - (k-1)); Infinity if no ordered chain exists. */
  protected static double orderedSpanSlack(List<int[]> posLists) {
    int k = posLists.size(); if (k < 2) return 0.0;
    int bestSpan = Integer.MAX_VALUE;
    int[] first = posLists.get(0);
    for (int p0 : first) {
      int start = p0, cur = p0; boolean ok = true;
      for (int i = 1; i < k; i++) {
        int[] L = posLists.get(i);
        int idx = Arrays.binarySearch(L, cur);
        int ins = (idx >= 0) ? idx : -idx - 1;
        if (ins >= L.length) { ok = false; break; }
        cur = L[ins];
      }
      if (ok) {
        int span = cur - start;
        if (span < bestSpan) bestSpan = span;
        if (bestSpan <= (k - 1)) break; // perfect adjacency
      }
    }
    if (bestSpan == Integer.MAX_VALUE) return Double.POSITIVE_INFINITY;
    return Math.max(0.0, bestSpan - (k - 1));
  }

  public int size() { return tokenHash.size(); }

  public void clear() {
    docRefs.clear();
    tokenHash.clear();
    posTokenHash.clear();
  }

  /** ========= Main: same flags & behavior as baseline ========= */
  public static void main(String[] args) {
    String dirName = args[args.length - 1];
    short docType = DocumentIterator.TYPE_TEXT;
    boolean stem = false, feedback = false;

    for (int i = 0; i < args.length - 1; i++) {
      String flag = args[i];
      if (flag.equals("-html")) docType = DocumentIterator.TYPE_HTML;
      else if (flag.equals("-stem")) stem = true;
      else if (flag.equals("-feedback")) feedback = true;
      else throw new IllegalArgumentException("Unknown flag: " + flag);
    }

    InvertedPosIndex index = new InvertedPosIndex(new File(dirName), docType, stem, feedback);
    index.processQueries();
  }
}
