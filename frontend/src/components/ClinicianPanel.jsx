// src/components/ClinicianPanel.jsx
// Decision support panel with recommendations and EHR summary.
// NEVER phrased as diagnosis.

import { motion } from 'framer-motion'
import { Stethoscope, AlertCircle, Info } from 'lucide-react'
import { riskLabel, severityClass } from '../lib/api'

const RECOMMENDATIONS = {
  low: [
    'Continue current monitoring protocol',
    'Reassess in morning if pattern persists',
    'Patient appears comfortable — normal sleep variation',
  ],
  medium: [
    'Increased monitoring frequency recommended',
    'Verify CPAP adherence and mask fit if prescribed',
    'Consider positional adjustment (semi-recumbent)',
    'Review recent sedative or alcohol use',
  ],
  high: [
    '⚡ Consider earlier CPAP titration review',
    '⚡ Evaluate supplemental oxygen indication',
    '⚡ Nurse call-bell within 15 min if clinically indicated',
    'Document desaturation event for morning attending review',
    'Review STOP-BANG score and escalation criteria',
  ],
}

export default function ClinicianPanel({ subject, riskScore }) {
  if (!subject) return null

  const level = riskScore < 0.35 ? 'low' : riskScore < 0.65 ? 'medium' : 'high'
  const recs = RECOMMENDATIONS[level]
  const levelColor = level === 'low' ? 'var(--risk-low)' : level === 'medium' ? 'var(--risk-medium)' : 'var(--risk-high)'

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* EHR Summary */}
      <div className="card-alt" style={{ padding: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
          <Stethoscope size={14} style={{ color: 'var(--primary)' }} />
          <h4 style={{ fontFamily: 'Space Grotesk', fontSize: '0.85rem' }}>
            Patient Profile (Synthetic EHR)
          </h4>
          <span style={{
            fontSize: '0.6rem', padding: '1px 7px',
            background: 'rgba(167,139,250,0.1)',
            color: 'var(--accent-lavender)',
            border: '1px solid rgba(167,139,250,0.25)',
            borderRadius: 999, marginLeft: 'auto',
          }}>
            Synthetic
          </span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px 16px', fontSize: '0.78rem' }}>
          {[
            ['Subject', subject.subject_id?.toUpperCase()],
            ['Sex', subject.sex],
            ['Age', subject.age ? `${subject.age} yrs` : '—'],
            ['BMI', subject.bmi ? `${subject.bmi}` : '—'],
            ['Prior AHI', subject.prior_ahi ? `${subject.prior_ahi.toFixed(1)}` : '—'],
            ['Hypertension', subject.hypertension ? 'Yes' : 'No'],
            ['STOP-BANG', subject.stopbang_score ?? '—'],
            ['ESS Score', subject.ess_score ?? '—'],
            ['CPAP Rx', subject.cpap_prescribed ? 'Yes' : 'No'],
          ].map(([label, val]) => (
            <div key={label}>
              <span style={{ color: 'var(--text-secondary)' }}>{label}: </span>
              <span style={{ color: 'var(--text-primary)', fontWeight: 500 }}>{val ?? '—'}</span>
            </div>
          ))}
        </div>

        <div style={{ marginTop: 12 }}>
          <span style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>OSA Severity: </span>
          <span className={`chip ${severityClass(subject.osa_severity)}`} style={{ fontSize: '0.68rem' }}>
            {subject.osa_severity?.toUpperCase()}
          </span>
        </div>
      </div>

      {/* Decision support */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
          <AlertCircle size={14} style={{ color: levelColor }} />
          <h4 style={{ fontFamily: 'Space Grotesk', fontSize: '0.85rem', color: levelColor }}>
            Clinical Decision Support ({riskLabel(riskScore)} Risk)
          </h4>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {recs.map((rec, i) => (
            <motion.div
              key={rec}
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.08 }}
              style={{
                padding: '8px 12px',
                borderRadius: 8,
                background: `${levelColor}10`,
                border: `1px solid ${levelColor}30`,
                fontSize: '0.78rem',
                color: 'var(--text-primary)',
                display: 'flex',
                alignItems: 'flex-start',
                gap: 8,
              }}
            >
              <div style={{
                width: 6, height: 6, borderRadius: '50%',
                background: levelColor, flexShrink: 0, marginTop: 5,
              }} />
              {rec}
            </motion.div>
          ))}
        </div>

        {/* Non-diagnostic disclaimer */}
        <div style={{
          marginTop: 12, padding: '8px 12px', borderRadius: 8,
          background: 'rgba(138,153,179,0.08)', border: '1px solid var(--border)',
          display: 'flex', gap: 8, alignItems: 'flex-start',
        }}>
          <Info size={12} style={{ color: 'var(--text-secondary)', flexShrink: 0, marginTop: 2 }} />
          <p style={{ fontSize: '0.68rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
            These are <strong>decision support suggestions</strong>, not diagnoses.
            Clinical judgement of a qualified clinician takes precedence at all times.
          </p>
        </div>
      </div>
    </div>
  )
}
