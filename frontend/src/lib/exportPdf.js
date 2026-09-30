/**
 * Screenshots a DOM element and saves it as a one-page A4 PDF.
 *
 * html2canvas and jsPDF are loaded on demand so they stay out of the main
 * bundle. Anything inside `element` marked `data-html2canvas-ignore` (buttons,
 * hints) is left out of the capture.
 */

function toDataUrl(url) {
  return fetch(url)
    .then((res) => res.blob())
    .then(
      (blob) =>
        new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result);
          reader.onerror = () => reject(reader.error);
          reader.readAsDataURL(blob);
        }),
    );
}

export async function exportElementToPdf(element, { filename, title }) {
  const [{ default: html2canvas }, { jsPDF }] = await Promise.all([
    import("html2canvas"),
    import("jspdf"),
  ]);

  // html2canvas renders an inline <svg> as a standalone image, and such an
  // image cannot load external <image href>s (signed URLs, blob: URLs). Inline
  // them as data URLs in the clone so the annotated photo survives the capture.
  const hrefs = new Set(
    [...element.querySelectorAll("svg image")]
      .map((node) => node.getAttribute("href"))
      .filter(Boolean),
  );
  const inlined = new Map(
    await Promise.all([...hrefs].map(async (href) => [href, await toDataUrl(href)])),
  );

  const canvas = await html2canvas(element, {
    scale: 2,
    backgroundColor: "#f1f5f9",
    logging: false,
    onclone: (doc) => {
      doc.querySelectorAll("svg image").forEach((node) => {
        const data = inlined.get(node.getAttribute("href"));
        if (data) node.setAttribute("href", data);
      });
    },
  });

  const pdf = new jsPDF({
    orientation: canvas.width >= canvas.height ? "landscape" : "portrait",
    unit: "mm",
    format: "a4",
  });
  const pageW = pdf.internal.pageSize.getWidth();
  const pageH = pdf.internal.pageSize.getHeight();
  const margin = 10;
  const headerH = 14;

  pdf.setFontSize(14);
  pdf.setTextColor(10, 37, 64);
  pdf.text(title, margin, margin + 5);
  pdf.setFontSize(9);
  pdf.setTextColor(120);
  pdf.text(`Generated ${new Date().toLocaleString()}`, margin, margin + 10);

  const scale = Math.min(
    (pageW - margin * 2) / canvas.width,
    (pageH - margin * 2 - headerH) / canvas.height,
  );
  const w = canvas.width * scale;
  const h = canvas.height * scale;
  pdf.addImage(
    canvas.toDataURL("image/jpeg", 0.92),
    "JPEG",
    (pageW - w) / 2,
    margin + headerH,
    w,
    h,
  );
  pdf.save(filename);
}
