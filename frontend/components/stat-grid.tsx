/** Dense definition-list grid for scalar readouts.
 *
 *  A hairline-divided grid of label/value cells: labels in .label-mono,
 *  values in ink (monospace + tabular when numeric). Not a chart — no
 *  color encodes magnitude here; the numbers are the display.
 */

export interface StatItem {
  label: string;
  value: string | number;
  hint?: string;
  /** Force monospace on/off; default: mono when the value contains a digit. */
  mono?: boolean;
}

export interface StatGridProps {
  items: StatItem[];
  className?: string;
}

export function StatGrid({ items, className = "" }: StatGridProps) {
  return (
    <dl
      className={`grid grid-cols-2 gap-px overflow-hidden rounded-md border border-bd bg-bd md:grid-cols-3 xl:grid-cols-4 ${className}`}
    >
      {items.map((item) => {
        const text = String(item.value);
        const mono = item.mono ?? /\d/.test(text);
        return (
          <div key={item.label} className="bg-surface px-3 py-2">
            <dt className="label-mono">{item.label}</dt>
            <dd
              className={`mt-0.5 text-[13px] text-ink ${mono ? "font-mono tabular" : ""}`}
            >
              {text}
            </dd>
            {item.hint && (
              <dd className="mt-0.5 text-[11px] leading-snug text-ink3">{item.hint}</dd>
            )}
          </div>
        );
      })}
    </dl>
  );
}
