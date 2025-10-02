function highlightLines(block) {
    if (block.dataset.highlighted) {
        return;
    }

    const linesToHighlight = block.getAttribute('hl_lines');
    const codeEl = block.querySelector('pre.highlight code');

    if (!linesToHighlight || !codeEl) return;

    const lineNumbersToHighlight = new Set();
    linesToHighlight.split(' ').forEach(range => {
        const parts = range.split('-').map(Number);
        if (parts.length === 1) {
            lineNumbersToHighlight.add(parts[0]);
        } else {
            for (let i = parts[0]; i <= parts[1]; i++) {
                lineNumbersToHighlight.add(i);
            }
        }
    });

    const newLineContainer = document.createElement('div');
    let currentLine = document.createElement('span');
    newLineContainer.appendChild(currentLine);

    Array.from(codeEl.childNodes).forEach(node => {
        const nodeLines = node.textContent.split('\n');
        nodeLines.forEach((lineText, i) => {
            if (i > 0) {
                // THE FIX: Add a newline character before starting the next line's span.
                newLineContainer.appendChild(document.createTextNode('\n'));
                currentLine = document.createElement('span');
                newLineContainer.appendChild(currentLine);
            }
            const newNode = node.cloneNode(false);
            newNode.textContent = lineText;
            currentLine.appendChild(newNode);
        });
    });

    Array.from(newLineContainer.childNodes).forEach((lineNode, i) => {
      // Line nodes are now spans and text nodes (\n), so we only check the spans.
      if (lineNode.nodeName === 'SPAN') {
        const lineNumber = Math.floor(i / 2) + 1;
        if (lineNumbersToHighlight.has(lineNumber)) {
          lineNode.classList.add('hll');
        }
      }
    });

    codeEl.innerHTML = newLineContainer.innerHTML;
    block.dataset.highlighted = 'true';
}

function highlightAllLines() {
    // Find all code blocks with the hl_lines attribute.
    const blocks = document.querySelectorAll('div[hl_lines]');
    blocks.forEach(highlightLines);
}
