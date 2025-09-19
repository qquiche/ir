package ir.vsr;

import java.io.*;

/**
 * A text document that does NOT remove stopwords or non-letters.
 * Used only for positional indexing (so distances reflect actual spacing).
 */
public class RawTextFileDocument extends TextFileDocument {
  public RawTextFileDocument(File file) {
    super(file, false); // stem=false, but we override prepareNextToken anyway
  }

  @Override
  protected void prepareNextToken() {
    // Just take the next candidate, lowercase, and return it — no stopword removal
    nextToken = getNextCandidateToken();
    if (nextToken == null) return;
    nextToken = nextToken.toLowerCase();
    // Optionally still filter out non-letters
    if (!allLetters(nextToken)) nextToken = null;
  }
}
