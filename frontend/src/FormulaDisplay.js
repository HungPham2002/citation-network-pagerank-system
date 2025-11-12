import React from 'react';
import './FormulaDisplay.css';

function FormulaDisplay({ algorithm }) {
  const formulas = {
    pagerank: {
      title: 'PageRank Algorithm',
      color: '#0047AB',
      mainFormula: 'PR(A) = (1-d)/N + d × Σ(PR(T_i)/C(T_i))',
      explanation: [
        { symbol: 'PR(A)', meaning: 'PageRank score of page A' },
        { symbol: 'd', meaning: 'Damping factor (typically 0.85)' },
        { symbol: 'N', meaning: 'Total number of pages in network' },
        { symbol: 'T_i', meaning: 'Pages that cite page A' },
        { symbol: 'C(T_i)', meaning: 'Number of outbound citations from T_i' }
      ],
      iterativeProcess: [
        '1. Initialize: PR(A) = 1/N for all pages',
        '2. For each iteration:',
        '   → Calculate new PR using formula',
        '   → Normalize scores to sum to 1',
        '3. Repeat until convergence (Δ < threshold)'
      ]
    },
    weighted_pagerank: {
      title: 'Weighted PageRank Algorithm',
      color: '#FF6B35',
      mainFormula: 'PR(A) = (1-d)/N + d × Σ(PR(T_i) × W(T_i,A))',
      explanation: [
        { symbol: 'W(T_i,A)', meaning: 'Weight of edge from T_i to A' },
        { symbol: 'Weight', meaning: '= 1 + (citations_A / max_citations)' },
        { symbol: 'd', meaning: 'Damping factor (typically 0.85)' },
        { symbol: 'PR(T_i)', meaning: 'PageRank of citing page' }
      ],
      iterativeProcess: [
        '1. Calculate edge weights based on citation counts',
        '2. Initialize: PR(A) = 1/N for all pages',
        '3. For each iteration:',
        '   → Apply weighted contributions',
        '   → Normalize scores',
        '4. Converge when Δ < threshold'
      ]
    },
    hits: {
      title: 'HITS Algorithm (Hubs & Authorities)',
      color: '#4CAF50',
      mainFormula: 'Authority(A) = Σ Hub(T_i), Hub(A) = Σ Authority(T_i)',
      explanation: [
        { symbol: 'Authority(A)', meaning: 'Authority score - how many cite A' },
        { symbol: 'Hub(A)', meaning: 'Hub score - how many A cites' },
        { symbol: 'T_i', meaning: 'Connected pages in citation network' },
        { symbol: 'Normalization', meaning: 'Divide by L2 norm each iteration' }
      ],
      iterativeProcess: [
        '1. Initialize: Auth(A) = Hub(A) = 1 for all',
        '2. For each iteration:',
        '   → Update Authority: Auth = A^T × Hub',
        '   → Update Hub: Hub = A × Auth',
        '   → Normalize both vectors (L2 norm)',
        '3. Converge when both Δ_auth and Δ_hub < threshold'
      ]
    }
  };

  const formula = formulas[algorithm];

  if (!formula) return null;

  return (
    <div className="formula-display-container" style={{ borderTop: `6px solid ${formula.color}` }}>
      <div className="formula-header">
        <span className="formula-icon">{formula.icon}</span>
        <h3 style={{ color: formula.color }}>{formula.title}</h3>
      </div>

      <div className="main-formula-box" style={{ background: `${formula.color}15` }}>
        <div className="formula-label">Main Formula:</div>
        <div className="formula-text">{formula.mainFormula}</div>
      </div>

      <div className="formula-explanation">
        <h4>Symbol Definitions:</h4>
        <table className="symbol-table">
          <tbody>
            {formula.explanation.map((item, idx) => (
              <tr key={idx}>
                <td className="symbol-cell">
                  <code>{item.symbol}</code>
                </td>
                <td className="meaning-cell">{item.meaning}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="iterative-process">
        <h4>🔄 Iterative Computation Process:</h4>
        <ol className="process-list">
          {formula.iterativeProcess.map((step, idx) => (
            <li key={idx} dangerouslySetInnerHTML={{ __html: step.replace(/→/g, '<span class="arrow">→</span>') }} />
          ))}
        </ol>
      </div>
    </div>
  );
}

export default FormulaDisplay;