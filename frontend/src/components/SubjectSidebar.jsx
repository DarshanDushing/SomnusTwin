// src/components/SubjectSidebar.jsx
// Subject selector with patient names, search, and severity color chips.
// Patient names are FICTITIOUS — shown for demo readability only.

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Search, ChevronRight, Moon, Wifi, WifiOff } from 'lucide-react'
import { severityClass } from '../lib/api'
import { getPatientName, getPatientDisplay } from '../lib/patientNames'

const SEVERITY_LABEL = {
  normal:   'Normal',
  mild:     'Mild OSA',
  moderate: 'Moderate OSA',
  severe:   'Severe OSA',
}

function SubjectRow({ subject, isSelected, onClick }) {
  const sev         = subject.osa_severity ?? 'normal'
  const patientName = getPatientName(subject.subject_id)
  const initials    = patientName.split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase()

  return (
    <motion.button
      initial={{ opacity: 0, x: -8 }}
      animate={{ opacity: 1, x: 0 }}
      whileHover={{ x: 2 }}
      onClick={onClick}
      id={`subject-${subject.subject_id}`}
      style={{
        width: '100%',
        textAlign: 'left',
        background: isSelected ? 'rgba(95,212,196,0.10)' : 'transparent',
        border: isSelected ? '1px solid rgba(95,212,196,0.25)' : '1px solid transparent',
        borderRadius: 10,
        padding: '10px 12px',
        cursor: 'pointer',
        transition: 'all 0.2s',
        display: 'flex',
        alignItems: 'center',
        gap: 10,
      }}
    >
      {/* Avatar initials */}
      <div style={{
        width: 34, height: 34, borderRadius: '50%',
        background: isSelected ? 'rgba(95,212,196,0.2)' : 'var(--surface-alt)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: '0.7rem', fontWeight: 700,
        color: isSelected ? 'var(--primary)' : 'var(--text-secondary)',
        flexShrink: 0, letterSpacing: '0.02em',
      }}>
        {initials}
      </div>

      {/* Info */}
      <div style={{ flex: 1, minWidth: 0 }}>
        {/* Patient name (primary) */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2 }}>
          <span style={{
            fontSize: '0.82rem', fontWeight: 600,
            color: 'var(--text-primary)',
            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          }}>
            {patientName}
          </span>
        </div>

        {/* Subject ID + severity badge (secondary) */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 3 }}>
          <span style={{ fontSize: '0.62rem', color: 'var(--text-secondary)', fontVariantNumeric: 'tabular-nums' }}>
            {subject.subject_id}
          </span>
          <span className={`chip ${severityClass(sev)}`} style={{ fontSize: '0.56rem', padding: '1px 5px' }}>
            {SEVERITY_LABEL[sev]}
          </span>
        </div>

        {/* Stats */}
        <div style={{ display: 'flex', gap: 8, fontSize: '0.65rem', color: 'var(--text-secondary)' }}>
          <span>AHI ≈{subject.ahi_approx?.toFixed(0)}</span>
          <span>·</span>
          <span>{subject.n_minutes}min</span>
          {subject.has_spo2 && <span style={{ color: 'var(--primary)' }}>· SpO₂✓</span>}
        </div>
      </div>

      <ChevronRight size={12} style={{ color: 'var(--text-secondary)', flexShrink: 0, opacity: isSelected ? 1 : 0.4 }} />
    </motion.button>
  )
}

export default function SubjectSidebar({ subjects = [], selectedId, onSelect, apiStatus }) {
  const [search, setSearch] = useState('')

  const filtered = subjects.filter(s =>
    s.subject_id.includes(search.toLowerCase()) ||
    s.osa_severity?.includes(search.toLowerCase()) ||
    getPatientName(s.subject_id).toLowerCase().includes(search.toLowerCase())
  )

  const counts = subjects.reduce((acc, s) => {
    acc[s.osa_severity] = (acc[s.osa_severity] ?? 0) + 1
    return acc
  }, {})

  return (
    <div className="sidebar" style={{ display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <div style={{ padding: '16px 16px 12px', borderBottom: '1px solid var(--border)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
          <Moon size={16} style={{ color: 'var(--primary)' }} />
          <span style={{ fontFamily: 'Space Grotesk', fontSize: '0.9rem', fontWeight: 600 }}>
            SomnusTwin
          </span>
        </div>

        {/* Search — searches by name or subject ID */}
        <div style={{ position: 'relative' }}>
          <Search size={13} style={{
            position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)',
            color: 'var(--text-secondary)',
          }} />
          <input
            id="subject-search"
            type="text"
            placeholder="Search by name or ID..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{
              width: '100%',
              background: 'var(--surface)',
              border: '1px solid var(--border)',
              borderRadius: 8,
              padding: '7px 10px 7px 30px',
              color: 'var(--text-primary)',
              fontSize: '0.78rem',
              outline: 'none',
            }}
          />
        </div>
      </div>

      {/* Severity summary chips */}
      {subjects.length > 0 && (
        <div style={{ padding: '8px 16px', borderBottom: '1px solid var(--border)' }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
            {Object.entries(counts).map(([sev, n]) => (
              <span key={sev} className={`chip ${severityClass(sev)}`} style={{ fontSize: '0.6rem' }}>
                {n} {SEVERITY_LABEL[sev]}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Subject list */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '8px 10px' }}>
        {filtered.length === 0 ? (
          <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', textAlign: 'center', padding: '24px 8px' }}>
            {subjects.length === 0
              ? 'No subjects. Run fetch_physionet.py'
              : 'No matches'}
          </p>
        ) : (
          <AnimatePresence>
            {filtered.map((s) => (
              <SubjectRow
                key={s.subject_id}
                subject={s}
                isSelected={s.subject_id === selectedId}
                onClick={() => onSelect(s.subject_id)}
              />
            ))}
          </AnimatePresence>
        )}
      </div>

      {/* Footer note — fictitious names disclaimer */}
      <div style={{ padding: '10px 16px', borderTop: '1px solid var(--border)', fontSize: '0.62rem', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
        {subjects.length} subjects · PhysioNet Apnea-ECG
        <br />
        <span style={{ opacity: 0.7 }}>Names fictitious — data real</span>
      </div>
    </div>
  )
}
