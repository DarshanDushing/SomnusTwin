// src/lib/patientNames.js
// Synthetic patient name mapping for demo readability.
// Names are FICTITIOUS — paired with real PhysioNet physiological recordings.
// Subject IDs (a01, a02, …) are real PhysioNet Apnea-ECG identifiers.
// Age/sex are derived from synthetic EHR profiles already generated.
//
// SINGLE SOURCE OF TRUTH — import getPatientName / getPatientDisplay everywhere.
// Never duplicate this map in individual components.

export const PATIENT_NAMES = {
  // a-series: severe OSA subjects
  a01: { name: 'Priya Sharma',     display: 'Priya Sharma, 44F'   },
  a02: { name: 'Meena Iyer',       display: 'Meena Iyer, 55F'     },
  a03: { name: 'Ramesh Kumar',     display: 'Ramesh Kumar, 67M'   },
  a04: { name: 'Anjali Nair',      display: 'Anjali Nair, 46F'    },
  a05: { name: 'Sunita Patel',     display: 'Sunita Patel, 57F'   },
  a06: { name: 'Mohan Das',        display: 'Mohan Das, 61M'      },
  a07: { name: 'Geeta Pillai',     display: 'Geeta Pillai, 52F'   },
  a08: { name: 'Suresh Varma',     display: 'Suresh Varma, 58M'   },
  a09: { name: 'Leela Krishnan',   display: 'Leela Krishnan, 63F' },
  a10: { name: 'Rajiv Mehta',      display: 'Rajiv Mehta, 54M'    },
  a11: { name: 'Usha Bhatt',       display: 'Usha Bhatt, 49F'     },
  a12: { name: 'Dinesh Sinha',     display: 'Dinesh Sinha, 70M'   },
  a13: { name: 'Pooja Agarwal',    display: 'Pooja Agarwal, 45F'  },
  a14: { name: 'Harish Gupta',     display: 'Harish Gupta, 60M'   },
  a15: { name: 'Shalini Desai',    display: 'Shalini Desai, 53F'  },
  a16: { name: 'Rakesh Tiwari',    display: 'Rakesh Tiwari, 66M'  },
  a17: { name: 'Kamla Chauhan',    display: 'Kamla Chauhan, 71F'  },
  a18: { name: 'Bharat Pandey',    display: 'Bharat Pandey, 47M'  },
  a19: { name: 'Sarla Mishra',     display: 'Sarla Mishra, 59F'   },
  a20: { name: 'Vinod Kapoor',     display: 'Vinod Kapoor, 64M'   },
  // b-series: borderline / mild
  b01: { name: 'Deepa Menon',      display: 'Deepa Menon, 31F'    },
  b02: { name: 'Kavya Reddy',      display: 'Kavya Reddy, 49F'    },
  b03: { name: 'Nikhil Verma',     display: 'Nikhil Verma, 38M'   },
  b04: { name: 'Archana Shetty',   display: 'Archana Shetty, 42F' },
  b05: { name: 'Sanjay Bose',      display: 'Sanjay Bose, 35M'    },
  // c-series: control / normal
  c01: { name: 'Vikram Rao',       display: 'Vikram Rao, 30M'     },
  c02: { name: 'Aditya Joshi',     display: 'Aditya Joshi, 37M'   },
  c03: { name: 'Nandita Shah',     display: 'Nandita Shah, 28F'   },
  c04: { name: 'Aryan Bhatia',     display: 'Aryan Bhatia, 33M'   },
  c05: { name: 'Ritu Malhotra',    display: 'Ritu Malhotra, 26F'  },
  // x-series: extra / mixed
  x01: { name: 'Arjun Singh',      display: 'Arjun Singh, 50M'    },
  x02: { name: 'Seema Rajan',      display: 'Seema Rajan, 43F'    },
  x03: { name: 'Tarun Nambiar',    display: 'Tarun Nambiar, 56M'  },
}

/**
 * Returns the short patient name (first + last name only).
 * Falls back to the raw subject_id if not found in the map.
 */
export function getPatientName(subjectId) {
  return PATIENT_NAMES[subjectId]?.name ?? subjectId
}

/**
 * Returns "Name, AgeS" display string (e.g. "Ramesh Kumar, 58M").
 * If a live EHR subject object is available (with .age and .sex), prefer those
 * over the hardcoded display string so they stay in sync with the API.
 * Falls back to subject_id if not found.
 */
export function getPatientDisplay(subjectId, subjectMeta = null) {
  const entry = PATIENT_NAMES[subjectId]
  if (!entry) return subjectId

  // If we have live EHR age/sex from the API, compose dynamically
  if (subjectMeta?.age != null && subjectMeta?.sex != null) {
    return `${entry.name}, ${subjectMeta.age}${subjectMeta.sex}`
  }
  return entry.display
}
