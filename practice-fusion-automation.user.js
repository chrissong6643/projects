// ==UserScript==
// @name         Practice Fusion EHR Auto-Fill
// @namespace    http://tampermonkey.net/
// @version      1.8.7
// @description  Automates form filling on Practice Fusion EHR charts
// @author       Medical Automation
// @match        *://*.practicefusion.com/*
// @match        *://static.practicefusion.com/*
// @match        https://static.practicefusion.com/apps/ehr/*
// @match        https://*.practicefusion.com/apps/ehr/*
// @include      *://*.practicefusion.com/*
// @include      *://static.practicefusion.com/*
// @noframes
// @grant        GM_log
// @grant        GM_setValue
// @grant        GM_getValue
// @grant        unsafeWindow
// @run-at       document-idle
// ==/UserScript==

(async function() {
  'use strict';

  // ============================================================================
  // CONFIGURATION & CONSTANTS
  // ============================================================================

  const CONFIG = {
    timeouts: {
      defaultWait: 2000,
      dropdownWait: 3000,
      templateWait: 5000,
      saveWait: 10000,
      shortWait: 500
    },
    formData: {
      chiefComplaint: 'Feeling well',
      subjectiveNote: 'no shortness of breath or pain',
      medicationReconciliation: 'Yes, reconciliation performed',
    }
  };

  const AUDIT_LOG = {
    startTime: new Date().toISOString(),
    steps: [],
    errors: []
  };

  let hasExecutedForThisPage = false;

  // ============================================================================
  // UTILITY FUNCTIONS
  // ============================================================================

  function log(message, data = null) {
    const timestamp = new Date().toISOString();
    const logEntry = `[${timestamp}] ${message}`;
    console.log(logEntry, data || '');
    
    AUDIT_LOG.steps.push({
      timestamp,
      message,
      data
    });
  }

  function logError(message, error) {
    console.error(`[ERROR] ${message}`, error);
    AUDIT_LOG.errors.push({
      timestamp: new Date().toISOString(),
      message,
      error: error.toString()
    });
  }

  async function wait(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  function isVisible(el) {
    if (!el) return false;
    const style = window.getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    return style.display !== 'none' && style.visibility !== 'hidden' && rect.width > 0 && rect.height > 0;
  }

  function textOf(el) {
    return (el?.textContent || '').replace(/\s+/g, ' ').trim();
  }

  function clickNative(el) {
    if (!el) return;
    el.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
    el.dispatchEvent(new MouseEvent('mouseup', { bubbles: true }));
    el.click();
  }

  function setFieldValue(el, value) {
    if (!el) return false;

    if (el.matches && (el.matches('input, textarea') || el.isContentEditable)) {
      el.focus();
      if (el.isContentEditable) {
        el.textContent = value;
      } else {
        el.value = value;
      }
      el.dispatchEvent(new Event('input', { bubbles: true }));
      el.dispatchEvent(new Event('change', { bubbles: true }));
      el.dispatchEvent(new Event('blur', { bubbles: true }));
      return true;
    }

    return false;
  }

  function findLabelElements(regex, scope = document) {
    return Array.from(scope.querySelectorAll('label,div,span,p,h1,h2,h3,h4')).filter(el => {
      const t = textOf(el);
      return t.length > 0 && regex.test(t) && isVisible(el);
    });
  }

  function findSectionByHeading(headingRegex) {
    const heading = Array.from(document.querySelectorAll('h1,h2,h3,h4,div,span')).find(el => {
      const t = textOf(el);
      return headingRegex.test(t) && isVisible(el);
    });
    if (!heading) return null;

    // Prefer a nearby ancestor that actually contains form controls.
    let section = heading.parentElement;
    for (let i = 0; i < 8 && section; i += 1) {
      const hasControl = section.querySelector('select, textarea, input, [role="combobox"], [role="listbox"]');
      if (hasControl) return section;
      section = section.parentElement;
    }
    return heading.parentElement || heading;
  }

  function scrollIntoViewIfNeeded(el) {
    if (!el) return;
    try {
      el.scrollIntoView({ behavior: 'instant', block: 'center' });
    } catch (_) {
      el.scrollIntoView(true);
    }
  }

  function setNativeSelectToYes(selectEl) {
    if (!selectEl) return false;
    const opt = Array.from(selectEl.options || []).find(o =>
      /yes\s*,?\s*reconciliation performed/i.test((o.text || '').trim()) || /^(yes)$/i.test((o.text || '').trim())
    );
    if (!opt) return false;
    selectEl.value = opt.value;
    selectEl.dispatchEvent(new Event('input', { bubbles: true }));
    selectEl.dispatchEvent(new Event('change', { bubbles: true }));
    return true;
  }

  function findLabelByExactText(labelText, scope = document, index = 0) {
    const normalizedTarget = labelText.replace(/\s+/g, ' ').trim().toLowerCase();
    const labels = Array.from(scope.querySelectorAll('label,div,span,p')).filter(el => {
      if (!isVisible(el)) return false;
      const t = textOf(el).toLowerCase();
      return t === normalizedTarget;
    });
    return labels[index] || null;
  }

  function findNativeSelectNearLabel(labelText, scope = document, index = 0) {
    const labelEl = findLabelByExactText(labelText, scope, index);
    if (!labelEl) return null;

    let container = labelEl.parentElement;
    for (let i = 0; i < 6 && container; i += 1) {
      const select = Array.from(container.querySelectorAll('select')).find(isVisible);
      if (select) return select;
      container = container.parentElement;
    }

    return null;
  }

  function findCustomDropdownControlNearLabel(labelRegex, scope = document, instanceIndex = 0) {
    const labels = findLabelElements(labelRegex, scope);
    const labelEl = labels[instanceIndex] || labels[0];
    if (!labelEl) return null;

    let container = labelEl;
    for (let i = 0; i < 6 && container; i += 1) {
      const nativeSelect = Array.from(container.querySelectorAll('select')).find(isVisible);
      if (nativeSelect) return nativeSelect;

      const candidates = Array.from(container.querySelectorAll(
        '[role="combobox"], [role="listbox"], button[aria-haspopup], button, div[aria-haspopup], input[readonly], input[aria-haspopup]'
      )).filter(isVisible);

      const strongCandidate = candidates.find(el => {
        const t = textOf(el);
        return /select\.\.\.|select|choose|reconciliation|completed|yes|no/i.test(t) ||
               el.hasAttribute('aria-expanded') ||
               el.getAttribute('role') === 'combobox';
      });

      if (strongCandidate) return strongCandidate;
      container = container.parentElement;
    }

    return null;
  }

  async function chooseVisibleOption(optionRegex, scope = document) {
    const start = Date.now();
    while (Date.now() - start < CONFIG.timeouts.dropdownWait + 2000) {
      const listboxes = Array.from(scope.querySelectorAll('[role="listbox"]')).filter(isVisible);
      const candidateListboxes = listboxes.length ? listboxes : Array.from(document.querySelectorAll('[role="listbox"]')).filter(isVisible);

      for (const lb of candidateListboxes) {
        const options = Array.from(lb.querySelectorAll('[role="option"]')).filter(isVisible);
        const match = options.find(opt => optionRegex.test(textOf(opt)));
        if (match) {
          clickNative(match);
          await wait(CONFIG.timeouts.shortWait);
          return true;
        }
      }

      await wait(120);
    }

    return false;
  }

  function getActiveTemplateListbox() {
    return Array.from(document.querySelectorAll('[role="listbox"]')).find(lb => {
      if (!isVisible(lb)) return false;
      const options = Array.from(lb.querySelectorAll('[role="option"]')).filter(isVisible);
      if (options.length === 0) return false;
      return !options.some(opt => /reconciliation performed|not applicable/i.test(textOf(opt)));
    }) || null;
  }

  async function clickTemplateEntry(preferredRegexes = []) {
    const listbox = getActiveTemplateListbox();
    if (!listbox) return false;

    const options = Array.from(listbox.querySelectorAll('[role="option"]')).filter(isVisible);
    if (options.length === 0) return false;

    let chosen = null;
    for (const rx of preferredRegexes) {
      chosen = options.find(opt => rx.test(textOf(opt)));
      if (chosen) break;
    }

    if (!chosen) {
      // Skip section-header rows such as "Subjective" / "Objective".
      chosen = options.find(opt => {
        const t = textOf(opt);
        if (!t) return false;
        if (/^(subjective|objective|assessment|plan)$/i.test(t)) return false;
        return t.length > 8;
      }) || options[0];
    }

    clickNative(chosen);
    await wait(CONFIG.timeouts.shortWait);
    return true;
  }

  async function setDropdownByLabel(labelRegex, optionRegex, instanceIndex = 0, scope = document) {
    // First try exact native select by label text when available.
    const explicitLabel = labelRegex
      .toString()
      .replace(/^\//, '')
      .replace(/\/[a-z]*$/i, '')
      .replace(/\\\?/g, '?')
      .replace(/\^|\$/g, '');

    const nativeSelect = findNativeSelectNearLabel(explicitLabel, scope, instanceIndex);
    if (nativeSelect) {
      const opt = Array.from(nativeSelect.options).find(o => optionRegex.test((o.text || '').trim()));
      if (!opt) {
        throw new Error(`Option not found in select for label: ${labelRegex}`);
      }
      nativeSelect.value = opt.value;
      nativeSelect.dispatchEvent(new Event('input', { bubbles: true }));
      nativeSelect.dispatchEvent(new Event('change', { bubbles: true }));
      await wait(CONFIG.timeouts.shortWait);
      return;
    }

    const control = findCustomDropdownControlNearLabel(labelRegex, scope, instanceIndex);
    if (!control) {
      throw new Error(`Dropdown not found for label: ${labelRegex}`);
    }

    if (control.tagName === 'SELECT') {
      const opt = Array.from(control.options).find(o => optionRegex.test((o.text || '').trim()));
      if (!opt) {
        throw new Error(`Option not found in select for label: ${labelRegex}`);
      }
      control.value = opt.value;
      control.dispatchEvent(new Event('change', { bubbles: true }));
      await wait(CONFIG.timeouts.shortWait);
      return;
    }

    clickNative(control);
    await wait(250);
    const ok = await chooseVisibleOption(optionRegex, control.parentElement || scope);
    if (ok) {
      return;
    }

    // Fallback for virtualized dropdowns where options are not text-queryable.
    control.focus();
    control.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true }));
    await wait(120);
    control.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    await wait(CONFIG.timeouts.shortWait);

    // Validate by checking nearby text changed away from "Select..." when possible.
    const controlText = textOf(control);
    const controlValue = control.value || '';
    if (/select\.\.\./i.test(controlText) && !/yes/i.test(controlText) && !/yes/i.test(controlValue)) {
      throw new Error(`Option not found in dropdown menu for label: ${labelRegex}`);
    }
  }

  function findDropdownButtonForLabel(labelRegex, scope = document) {
    // Use short text nodes first to avoid matching giant container divs.
    const labels = Array.from(scope.querySelectorAll('label,span,p,div')).filter(el => {
      if (!isVisible(el)) return false;
      const t = textOf(el);
      if (!labelRegex.test(t)) return false;
      return t.length <= 120;
    }).sort((a, b) => textOf(a).length - textOf(b).length);

    for (const labelEl of labels) {
      const labelRect = labelEl.getBoundingClientRect();

      let container = labelEl.parentElement;
      for (let i = 0; i < 4 && container; i += 1) {
        const btnCandidates = Array.from(container.querySelectorAll('button,[role="combobox"],[aria-haspopup="listbox"]')).filter(el => {
          if (!isVisible(el)) return false;
          const t = textOf(el);
          if (!/select\.\.\.|yes|no|not applicable|reconciliation/i.test(t)) return false;

          // Keep only controls on the same visual row (or very close) to the label.
          const r = el.getBoundingClientRect();
          const sameRow = Math.abs(r.top - labelRect.top) < 100;
          const nearRow = Math.abs((r.top + r.height / 2) - (labelRect.top + labelRect.height / 2)) < 110;
          return sameRow || nearRow;
        });

        if (btnCandidates.length > 0) {
          // Prefer Select... control first when available.
          const selectBtn = btnCandidates.find(el => /select\.\.\./i.test(textOf(el)));
          return selectBtn || btnCandidates[0];
        }

        container = container.parentElement;
      }
    }
    return null;
  }

  function findEncounterSectionByHeading(headingRegex, requiredTextRegex) {
    const heading = Array.from(document.querySelectorAll('h1,h2,h3,h4,div,span')).find(el => {
      return headingRegex.test(textOf(el)) && isVisible(el);
    });
    if (!heading) return null;

    let container = heading.parentElement;
    for (let i = 0; i < 8 && container; i += 1) {
      const t = textOf(container);
      if (requiredTextRegex.test(t)) return container;
      container = container.parentElement;
    }
    return null;
  }

  function findEditorForNoteHeading(noteHeadingRegex) {
    const heading = Array.from(document.querySelectorAll('h3,h4,div,span')).find(el => {
      return noteHeadingRegex.test(textOf(el)) && isVisible(el);
    });

    const editors = Array.from(document.querySelectorAll('textarea,[contenteditable="true"],[role="textbox"]')).filter(isVisible);
    if (editors.length === 0) return null;
    if (!heading) return editors[0];

    if (heading.parentElement) {
      const localEditor = Array.from(heading.parentElement.querySelectorAll('textarea,[contenteditable="true"],[role="textbox"]')).find(isVisible);
      if (localEditor) return localEditor;
    }

    const hr = heading.getBoundingClientRect();
    const nearby = editors
      .map(ed => ({ ed, r: ed.getBoundingClientRect() }))
      .filter(({ r }) => r.top >= hr.top - 24 && r.top <= hr.top + 520 && Math.abs(r.left - hr.left) < 220)
      .sort((a, b) => a.r.top - b.r.top);

    if (nearby.length > 0) return nearby[0].ed;

    // Fallback: closest editor vertically below the heading.
    const byDistance = editors
      .map(ed => ({ ed, d: Math.abs(ed.getBoundingClientRect().top - hr.top) }))
      .sort((a, b) => a.d - b.d);
    return byDistance[0]?.ed || null;
  }

  function activateNoteEditor(noteHeadingRegex, sectionHeadingRegex = null) {
    let editor = findEditorForNoteHeading(noteHeadingRegex);

    if (!editor && sectionHeadingRegex) {
      const section = findSectionByHeading(sectionHeadingRegex);
      if (section) {
        editor = Array.from(section.querySelectorAll('textarea,[contenteditable="true"],[role="textbox"]')).find(isVisible) || null;
      }
    }

    if (!editor) return false;

    scrollIntoViewIfNeeded(editor);
    try {
      editor.focus();
      clickNative(editor);
      editor.dispatchEvent(new Event('focus', { bubbles: true }));
    } catch (_) {
      // Best effort; caller validates result through downstream behavior.
    }

    return true;
  }

  function isOnTargetChartPage() {
    const url = window.location.href;
    return /\/PF\/charts\/patients/i.test(url) || /\/charts\/patients/i.test(url);
  }

  function hasChartMarkers() {
    const bodyText = (document.body && document.body.textContent) || '';
    return /chief complaint|quality of care|assessment|subjective|objective/i.test(bodyText);
  }

  function findByText(text, selector = '*') {
    const regex = new RegExp(text, 'i');
    const elements = document.querySelectorAll(selector);
    for (const el of elements) {
      if (regex.test(el.textContent) && el.offsetParent !== null) {
        return el;
      }
    }
    return null;
  }

  function findByLabelText(labelText) {
    const regex = new RegExp(labelText, 'i');
    // Look for label element that matches, then get associated input
    const labels = document.querySelectorAll('label');
    for (const label of labels) {
      if (regex.test(label.textContent)) {
        if (label.htmlFor) {
          return document.getElementById(label.htmlFor);
        }
        return label.querySelector('input, textarea, select');
      }
    }
    return null;
  }

  async function waitForElement(selector, timeout = CONFIG.timeouts.defaultWait) {
    const startTime = Date.now();
    while (Date.now() - startTime < timeout) {
      const el = document.querySelector(selector);
      if (el && el.offsetParent !== null) return el;
      await wait(100);
    }
    return null;
  }

  // ============================================================================
  // FIELD FILL FUNCTIONS
  // ============================================================================

  async function fillChiefComplaint() {
    try {
      log('Step 1: Filling Chief Complaint...');
      
      // Strategy 1: Look for textarea with data-testid or aria-label
      let input = document.querySelector('textarea[data-testid*="chief" i]') ||
          document.querySelector('textarea[aria-label*="chief" i]') ||
          document.querySelector('textarea[placeholder*="problems or symptoms" i]') ||
          document.querySelector('input[placeholder*="problems or symptoms" i]');
      
      // Strategy 2: Find by label text + adjacent textarea
      if (!input) {
        const label = Array.from(document.querySelectorAll('label, div')).find(el => 
          /chief complaint/i.test(el.textContent)
        );
        if (label) {
          input = label.parentElement?.querySelector('textarea, input[type="text"], input:not([type])');
        }
      }

      // Strategy 3: textbox with "Chief complaint" aria label
      if (!input) {
        input = Array.from(document.querySelectorAll('input,textarea,[contenteditable="true"]')).find(el =>
          isVisible(el) && /chief complaint/i.test(el.getAttribute('aria-label') || '')
        );
      }

      // Strategy 4: First visible real text input or textarea in the encounter editor
      if (!input) {
        input = Array.from(document.querySelectorAll('textarea, input[type="text"], input:not([type]), [contenteditable="true"]')).find(isVisible);
      }

      if (!input) {
        throw new Error('Chief Complaint textarea not found');
      }

      const filled = setFieldValue(input, CONFIG.formData.chiefComplaint);
      if (!filled) {
        throw new Error('Chief Complaint target was not a writable field');
      }
      
      await wait(CONFIG.timeouts.shortWait);
      log('✓ Chief Complaint filled: ' + CONFIG.formData.chiefComplaint);
      
    } catch (error) {
      logError('Chief Complaint fill failed', error);
      throw error;
    }
  }

  async function selectMedicationReconciliation() {
    try {
      log('Step 2: Setting Medication Reconciliation dropdown...');

      // Use the Encounter Medications panel (contains reconciliation label and med table headers).
      const medicationsSection =
        findEncounterSectionByHeading(/^Medications$/i, /was medication reconciliation completed\?|name\s+sig\s+start/i) ||
        Array.from(document.querySelectorAll('div,section')).find(el => {
          const t = textOf(el);
          return /was medication reconciliation completed\?/i.test(t) && /name\s+sig\s+start/i.test(t);
        }) ||
        document;
      scrollIntoViewIfNeeded(medicationsSection);

      const medButton = findDropdownButtonForLabel(/WAS MEDICATION RECONCILIATION COMPLETED\?/i, medicationsSection);
      if (!medButton) {
        throw new Error('Medication reconciliation dropdown button not found');
      }

      clickNative(medButton);
      await wait(200);
      const ok = await chooseVisibleOption(/^yes\s*,?\s*reconciliation performed$/i, medicationsSection);
      if (!ok) {
        throw new Error('Medication reconciliation option "Yes, reconciliation performed" not found');
      }

      await wait(CONFIG.timeouts.shortWait);
      log('✓ Medication Reconciliation set to Yes, reconciliation performed');
      
    } catch (error) {
      logError('Medication Reconciliation selection failed', error);
      throw error;
    }
  }

  // Find the "View templates" button that is a sibling of the specific note heading (h3/h4).
  function findViewTemplatesNearNoteHeading(noteLabelRegex) {
    const noteHeading = Array.from(document.querySelectorAll('h3,h4')).find(el =>
      noteLabelRegex.test(textOf(el)) && isVisible(el)
    );
    if (!noteHeading || !noteHeading.parentElement) return null;
    return Array.from(noteHeading.parentElement.querySelectorAll('button,a,span')).find(el =>
      /^view templates$/i.test(textOf(el)) && isVisible(el)
    ) || null;
  }

  function findAncestorWithVisibleListbox(startEl) {
    let container = startEl;
    for (let i = 0; i < 8 && container; i += 1) {
      const listbox = Array.from(container.querySelectorAll('[role="listbox"]')).find(isVisible);
      if (listbox) return container;
      container = container.parentElement;
    }
    return null;
  }

  function getTemplatePanelListbox() {
    const searchBox = Array.from(document.querySelectorAll('input,textarea,[role="textbox"]')).find(el => {
      if (!isVisible(el)) return false;
      const label = `${el.getAttribute('aria-label') || ''} ${el.getAttribute('placeholder') || ''} ${textOf(el)}`;
      return /search all templates and items/i.test(label);
    });

    if (searchBox) {
      const container = findAncestorWithVisibleListbox(searchBox);
      if (container) {
        return Array.from(container.querySelectorAll('[role="listbox"]')).find(isVisible) || null;
      }
    }

    const backBtn = Array.from(document.querySelectorAll('button,a,span')).find(el =>
      (/^<\s*back$/i.test(textOf(el)) || /^back$/i.test(textOf(el))) && isVisible(el)
    );

    if (backBtn) {
      const container = findAncestorWithVisibleListbox(backBtn);
      if (container) {
        return Array.from(container.querySelectorAll('[role="listbox"]')).find(isVisible) || null;
      }
    }

    return getActiveTemplateListbox();
  }

  function findFirstRealTemplateOption(listbox) {
    if (!listbox) return null;

    const options = Array.from(listbox.querySelectorAll('[role="option"]')).filter(isVisible);
    return options.find(opt => {
      const t = textOf(opt);
      if (!t) return false;
      const sectionHeading = opt.querySelector('h1,h2,h3,h4,h5,h6');
      const hasParagraphContent = opt.querySelector('p');
      if (sectionHeading && !hasParagraphContent && /^(subjective|objective|assessment|plan)$/i.test(textOf(sectionHeading))) {
        return false;
      }
      if (/^no template items found$/i.test(t)) return false;
      return true;
    }) || null;
  }

  async function fillSubjectiveNote() {
    try {
      log('Step 3: Filling Subjective Note...');

      const subjectiveSection = findSectionByHeading(/^Subjective$/i) || document;
      activateNoteEditor(/^subjective note$/i, /^Subjective$/i);

      // Anchor on "Subjective note" h4 heading's parent to find the correct View templates button.
      const subjViewTemplatesLink = findViewTemplatesNearNoteHeading(/^subjective note$/i);

      if (!subjViewTemplatesLink) {
        throw new Error('Subjective "View templates" control not found');
      }
      clickNative(subjViewTemplatesLink);

      await wait(CONFIG.timeouts.templateWait);

      // Select template from the active template sidebar only.
      const templateListbox = getTemplatePanelListbox();

      if (!templateListbox) {
        throw new Error('Template listbox not found for Subjective templates');
      }

      const templateOptions = Array.from(templateListbox.querySelectorAll('[role="option"]')).filter(isVisible);
      if (templateOptions.length === 0) {
        throw new Error('No template options available in Subjective templates');
      }

      // Select the "New template" first (required workflow), fallback to first template.
      const newTemplate = templateOptions.find(opt => /^new template\b/i.test(textOf(opt)));
      clickNative(newTemplate || templateOptions[0]);
      await wait(CONFIG.timeouts.templateWait);

      // Ensure insertion target stays on Subjective before selecting entry text.
      activateNoteEditor(/^subjective note$/i, /^Subjective$/i);

      const subjectiveDetailListbox = getTemplatePanelListbox();
      const subjectiveEntry = Array.from((subjectiveDetailListbox || templateListbox).querySelectorAll('[role="option"]')).filter(isVisible)
        .find(opt => /no shortness of breath or pain\.?/i.test(textOf(opt))) ||
        findFirstRealTemplateOption(subjectiveDetailListbox || templateListbox);

      if (!subjectiveEntry) {
        throw new Error('Subjective template entry not found');
      }

      clickNative(subjectiveEntry);
      await wait(CONFIG.timeouts.shortWait);

      // Now find the Subjective editor and add the note text
      let editor = findEditorForNoteHeading(/^subjective note$/i) ||
        Array.from(subjectiveSection.querySelectorAll('textarea,[contenteditable="true"],[role="textbox"]')).find(isVisible);

      if (!editor) {
        throw new Error('Subjective Note textarea not found');
      }

      // Keep text write as a fallback only if template entry did not populate it.
      if (!textOf(editor)) {
        const wrote = setFieldValue(editor, CONFIG.formData.subjectiveNote);
        if (!wrote) {
          throw new Error('Subjective Note textarea is not writable');
        }
      }
      
      await wait(CONFIG.timeouts.shortWait);
      log('✓ Subjective template selected and note filled: ' + CONFIG.formData.subjectiveNote);
      
    } catch (error) {
      logError('Subjective Note fill failed', error);
      // Non-critical, continue
      log('⚠ Subjective Note skipped - continuing with Objective');
    }
  }

  async function fillObjectiveFromTemplate() {
    try {
      log('Step 4: Filling Objective section from ROS+PE template...');

      const getActiveTabs = () => Array.from(document.querySelectorAll('button')).filter(el => isVisible(el) && /summary|templates|past encounters/i.test(textOf(el))).map(textOf).filter(t => /summary|templates|past encounters/i.test(t));

      // Open the Objective note's View templates panel.
      activateNoteEditor(/^objective note$/i, /^Objective$/i);
      log('DEBUG: [1] Tabs before View templates click', { tabs: getActiveTabs() });

      const viewTemplatesLink = findViewTemplatesNearNoteHeading(/^objective note$/i);
      if (!viewTemplatesLink) {
        throw new Error('Objective "View templates" control not found');
      }
      clickNative(viewTemplatesLink);
      await wait(CONFIG.timeouts.templateWait);
      log('DEBUG: [2] Tabs after View templates click', { tabs: getActiveTabs() });

      const targetPeRegex = /^PE:\s*VS:\s*above\s*GEN:\s*AAOX3\.\s*No\s*acute\s*distress\.\s*Neck:\s*supple,\s*no\s*JVD\s*or\s*bruits\.\s*Lungs:\s*CTA-B\.\s*Heart:\s*RRR,\s*no\s*m\/r\/g\.\s*Abd:\s*Soft,\s*BS\(\+\)\.\s*Mild\s*epigastric\s*tenderness\.\s*No\s*CVA\.\s*Ext:\s*No\s*edema\.\s*NT\.?/i;

      const findTargetPeEntryInPanel = () => {
        const lb = getTemplatePanelListbox();
        if (!lb) return null;
        return Array.from(lb.querySelectorAll('[role="option"]')).filter(isVisible).find(opt =>
          targetPeRegex.test(textOf(opt))
        ) || null;
      };

      // If ROS+PE detail is already open, click the requested PE option directly.
      let targetPeEntry = findTargetPeEntryInPanel();
      log('DEBUG: [3] Direct target PE lookup', { found: !!targetPeEntry, tabs: getActiveTabs() });
      if (targetPeEntry) {
        clickNative(targetPeEntry);
        await wait(CONFIG.timeouts.shortWait);
        log('DEBUG: [4] After direct PE click', { tabs: getActiveTabs() });
        log('✓ Objective ROS+PE target PE entry clicked');
        return;
      }

      // If we landed inside a template detail view, go back to the template list first.
      const templateSearchBox = Array.from(document.querySelectorAll('input,textarea,[role="textbox"]')).find(el => {
        if (!isVisible(el)) return false;
        const label = `${el.getAttribute('aria-label') || ''} ${el.getAttribute('placeholder') || ''} ${textOf(el)}`;
        return /search all templates and items/i.test(label);
      });
      const backToList = Array.from(document.querySelectorAll('button,a,span')).find(el =>
        (/^<\s*back$/i.test(textOf(el)) || /^back$/i.test(textOf(el))) && isVisible(el)
      );
      log('DEBUG: [5] Back button check', { searchBoxFound: !!templateSearchBox, backBtnFound: !!backToList, tabs: getActiveTabs() });
      if (!templateSearchBox && backToList) {
        log('DEBUG: [6] Clicking back button');
        clickNative(backToList);
        await wait(350);
        log('DEBUG: [7] After back click', { tabs: getActiveTabs() });
      }

      const templateListbox = getTemplatePanelListbox();
      if (!templateListbox) {
        throw new Error('Objective templates listbox not found');
      }
      log('DEBUG: [8] Template listbox found', { tabs: getActiveTabs() });

      // Click the "ROS+PE" template option directly in the template sidebar.
      const rospeOption = Array.from(templateListbox.querySelectorAll('[role="option"]')).filter(isVisible)
        .find(opt => /ros\+pe/i.test(textOf(opt)));
      if (!rospeOption) {
        throw new Error('"ROS+PE" template not found in Objective templates list');
      }
      log('DEBUG: [9] ROS+PE option found, about to click', { tabs: getActiveTabs() });
      clickNative(rospeOption);
      await wait(CONFIG.timeouts.templateWait);
      log('DEBUG: [10] After ROS+PE click', { tabs: getActiveTabs() });

      const objectiveDetailListbox = getTemplatePanelListbox();
      const firstEntry = Array.from((objectiveDetailListbox || document).querySelectorAll('[role="option"]')).filter(isVisible)
        .find(opt => targetPeRegex.test(textOf(opt))) ||
        findFirstRealTemplateOption(objectiveDetailListbox);
      if (!firstEntry) {
        throw new Error('ROS+PE template has no selectable entry items');
      }
      log('DEBUG: [11] First entry found, about to click', { tabs: getActiveTabs(), entryText: textOf(firstEntry).slice(0, 100) });
      clickNative(firstEntry);
      await wait(CONFIG.timeouts.shortWait);
      log('DEBUG: [12] After entry click', { tabs: getActiveTabs() });

      log('✓ Objective ROS+PE target entry clicked');

    } catch (error) {
      logError('Objective template fill failed', error);
      throw error;
    }
  }

  async function fillAssessmentFromPastEncounter() {
    try {
      log('Step 5: Filling Assessment from past encounters...');

      // Ensure paste target is Assessment note before opening import panel.
      activateNoteEditor(/^assessment note$/i, /^Assessment$/i);

      // In Assessment section, use explicit "Import past encounter" control.
      const assessmentSection = findSectionByHeading(/^Assessment$/i) || document;
      const importPastEncounterLink = Array.from(assessmentSection.querySelectorAll('a,button,span')).find(el =>
        /^import past encounter$/i.test(textOf(el)) && isVisible(el)
      );

      if (!importPastEncounterLink) {
        throw new Error('Assessment "Import past encounter" control not found');
      }

      clickNative(importPastEncounterLink);
      await wait(CONFIG.timeouts.templateWait);
      log('DEBUG: [A1] After Import past encounter click');

      // Click Import part of note from the first encounter row.
      const findAssessmentDetailEntry = async (listbox) => {
        let detailEntry = null;
        let currentSection = '';

        try {
          listbox.scrollTop = 0;
        } catch (_) {
          // Best effort only.
        }

        for (let pass = 0; pass < 40 && !detailEntry; pass += 1) {
          const detailOptions = Array.from(listbox.querySelectorAll('[role="option"]')).filter(isVisible);
          for (const opt of detailOptions) {
            const t = textOf(opt);
            if (/^(subjective|objective|assessment|plan)$/i.test(t)) {
              currentSection = t.toLowerCase();
              continue;
            }
            if (currentSection === 'assessment' && t.length > 0) {
              detailEntry = opt;
              break;
            }
          }

          if (detailEntry) break;

          const before = listbox.scrollTop;
          listbox.scrollTop += Math.max(120, Math.floor(listbox.clientHeight * 0.7));
          await wait(80);
          if (listbox.scrollTop === before) break;
        }

        return detailEntry;
      };

      const getEncounterRows = () => Array.from(document.querySelectorAll('li,[role="option"],div')).filter(el => {
        if (!isVisible(el)) return false;
        return /\d{2}\/\d{2}\/\d{2}:\s*office visit,\s*soap note/i.test(textOf(el));
      });

      const getEncounterListbox = () => Array.from(document.querySelectorAll('[role="listbox"]')).find(lb => {
        if (!isVisible(lb)) return false;
        return /import part of note/i.test(textOf(lb));
      }) || null;

      // Scan and scroll through encounter rows until we find one with Assessment detail entries.
      let detailEntry = null;
      let selectedRowSummary = '';
      const visitedRows = new Set();
      for (let pass = 0; pass < 20 && !detailEntry; pass += 1) {
        const rows = getEncounterRows();
        for (const row of rows) {
          const rowSummary = textOf(row).slice(0, 120);
          if (!rowSummary || visitedRows.has(rowSummary)) continue;
          visitedRows.add(rowSummary);

          const importPartBtn = Array.from(row.querySelectorAll('a,button,span')).find(el =>
            /^import part of note$/i.test(textOf(el)) && isVisible(el)
          );
          if (!importPartBtn) continue;

          selectedRowSummary = rowSummary;
          clickNative(importPartBtn);
          await wait(CONFIG.timeouts.templateWait);
          log('DEBUG: [A2] Clicked Import part of note', { pass, rowSummary: selectedRowSummary });

          const detailListbox = getTemplatePanelListbox();
          if (!detailListbox) {
            throw new Error('Past encounter detail listbox not found after Import part of note');
          }

          detailEntry = await findAssessmentDetailEntry(detailListbox);
          if (!detailEntry) {
            // Fallback for non-sectioned rows that still mention assessment.
            detailEntry = Array.from(detailListbox.querySelectorAll('[role="option"]')).filter(isVisible)
              .find(opt => /assessment/i.test(textOf(opt)) && !/^assessment$/i.test(textOf(opt))) || null;
          }

          if (detailEntry) break;

          const backBtn = Array.from(document.querySelectorAll('button,a,span')).find(el =>
            (/^<\s*back$/i.test(textOf(el)) || /^back$/i.test(textOf(el))) && isVisible(el)
          );
          if (!backBtn) break;
          clickNative(backBtn);
          await wait(350);
        }

        if (detailEntry) break;

        const encounterListbox = getEncounterListbox();
        if (!encounterListbox) break;
        const before = encounterListbox.scrollTop;
        encounterListbox.scrollTop += Math.max(140, Math.floor(encounterListbox.clientHeight * 0.75));
        await wait(120);
        if (encounterListbox.scrollTop === before) break;
      }

      if (!detailEntry) {
        const fallbackFullImportBtn = getEncounterRows()
          .map(row => Array.from(row.querySelectorAll('a,button,span')).find(el =>
            /^import full note$/i.test(textOf(el)) && isVisible(el)
          ))
          .find(Boolean) || null;

        if (fallbackFullImportBtn) {
          activateNoteEditor(/^assessment note$/i, /^Assessment$/i);
          clickNative(fallbackFullImportBtn);
          await wait(250);

          const confirmImportBtn = Array.from(document.querySelectorAll('button,a,span')).find(el =>
            /^import$/i.test(textOf(el)) && isVisible(el)
          );
          if (confirmImportBtn) {
            clickNative(confirmImportBtn);
            await wait(CONFIG.timeouts.templateWait);
            log('DEBUG: [A3] Confirmed full-note import modal');
          } else {
            await wait(CONFIG.timeouts.templateWait);
          }

          log('DEBUG: [A3] Fallback used Import full note for Assessment');
          log('✓ Assessment imported from past encounter (full note fallback)');
          return;
        }

        throw new Error('No Assessment text entries found in past encounter detail list');
      }

      // Re-activate Assessment editor to avoid inserting into the wrong note.
      activateNoteEditor(/^assessment note$/i, /^Assessment$/i);
      clickNative(detailEntry);
      await wait(CONFIG.timeouts.shortWait);
      log('DEBUG: [A3] Clicked detail text item', { itemText: textOf(detailEntry).slice(0, 80), sourceRow: selectedRowSummary });
      log('✓ Assessment imported from past encounter');
      
    } catch (error) {
      logError('Assessment import failed', error);
      throw error;
    }
  }

  async function setQualityOfCareControls() {
    try {
      log('Step 6: Setting Quality of Care section...');

      // Find Quality of Care section with its reconciliation labels.
      const qualitySection =
        findEncounterSectionByHeading(/^Quality of care$/i, /was diagnoses reconciliation completed\?|was medication allergy reconciliation completed\?|was medication reconciliation completed\?/i) ||
        Array.from(document.querySelectorAll('div, section')).find(el => {
          const t = textOf(el);
          return /quality of care/i.test(t) && /was diagnoses reconciliation completed\?/i.test(t);
        });

      if (!qualitySection) {
        throw new Error('Quality of Care section not found');
      }

      scrollIntoViewIfNeeded(qualitySection);

      // Only set the first 2 dropdowns to "Yes" (not all 3)
      const labels = [
        /WAS DIAGNOSES RECONCILIATION COMPLETED\?/i,
        /WAS MEDICATION ALLERGY RECONCILIATION COMPLETED\?/i
      ];

      for (const label of labels) {
        const button = findDropdownButtonForLabel(label, qualitySection);
        if (!button) {
          throw new Error(`Quality of Care dropdown button not found for ${label}`);
        }
        clickNative(button);
        await wait(160);
        const ok = await chooseVisibleOption(/^yes\s*,?\s*reconciliation performed$/i, qualitySection);
        if (!ok) {
          throw new Error(`Quality of Care yes option not found for ${label}`);
        }
      }

      log('✓ Set 2 quality dropdowns to Yes, reconciliation performed');

      // Check first 4 checkboxes
      const checkboxes = qualitySection.querySelectorAll('input[type="checkbox"], [role="checkbox"]');
      let checkboxesChecked = 0;
      
      const checkboxesToCheck = Math.min(4, checkboxes.length);
      for (let i = 0; i < checkboxesToCheck; i++) {
        const checkbox = checkboxes[i];
        if (checkbox.getAttribute('type') === 'checkbox') {
          if (!checkbox.checked) {
            checkbox.click();
            checkboxesChecked++;
          }
        } else {
          // ARIA checkbox
          const isChecked = checkbox.getAttribute('aria-checked') === 'true';
          if (!isChecked) {
            checkbox.click();
            checkboxesChecked++;
          }
        }
        await wait(CONFIG.timeouts.shortWait);
      }

      log(`✓ Checked ${checkboxesChecked} checkboxes (first 4)`);
      
    } catch (error) {
      logError('Quality of Care controls failed', error);
      throw error;
    }
  }

  async function clickSaveButton() {
    try {
      log('Step 7: Clicking Save button...');

      // Find Save button - usually top right, look for text "Save"
      const saveBtn = Array.from(document.querySelectorAll('button')).find(btn =>
        /^save$/i.test(btn.textContent?.trim()) &&
        btn.offsetParent !== null
      );

      if (!saveBtn) {
        throw new Error('Save button not found');
      }

      log('Found Save button, clicking it...');
      saveBtn.click();

      // Wait for confirmation
      await wait(CONFIG.timeouts.saveWait);

      // Check for success message
      const pageText = document.body.textContent;
      const successIndicators = ['success', 'saved', 'completed', 'saved successfully'];
      const isSuccessful = successIndicators.some(indicator =>
        new RegExp(indicator, 'i').test(pageText)
      );

      if (isSuccessful) {
        log('✓ Form saved successfully');
      } else {
        log('⚠ Save clicked, verifying page state...');
      }

      // Store audit log
      GM_setValue('lastAuditLog', JSON.stringify(AUDIT_LOG));
      
    } catch (error) {
      logError('Save button click failed', error);
      throw error;
    }
  }

  // ============================================================================
  // MAIN EXECUTION
  // ============================================================================

  async function executeFormAutomation() {
    const failures = [];

    async function runStep(name, fn) {
      try {
        await fn();
      } catch (error) {
        failures.push(`${name}: ${error.message}`);
      }
    }

    try {
      log('================== STARTING FORM AUTOMATION ==================');

      // Phase 1: Chief Complaint
      await runStep('Chief Complaint', fillChiefComplaint);
      await wait(500);

      // Phase 2: Medication Reconciliation
      await runStep('Medication Reconciliation', selectMedicationReconciliation);
      await wait(500);

      // Phase 3: Subjective Note
      await runStep('Subjective Note', fillSubjectiveNote);
      await wait(500);

      // Phase 4: Objective from Template
      await runStep('Objective Template', fillObjectiveFromTemplate);
      await wait(500);

      // Phase 5: Assessment from Past Encounter
      await runStep('Assessment Import', fillAssessmentFromPastEncounter);
      await wait(500);

      // Phase 6: Quality of Care
      await runStep('Quality of Care', setQualityOfCareControls);
      await wait(500);

      if (failures.length > 0) {
        logError('Automation finished with step failures', new Error(failures.join(' | ')));
        alert(`⚠ Automation completed with issues and did NOT click Save.\n\n${failures.join('\n')}`);
        return;
      }

      // Phase 7: Save only if all prior steps succeeded.
      await runStep('Save', clickSaveButton);

      if (failures.length > 0) {
        alert(`⚠ Form updates ran, but Save failed.\n\n${failures.join('\n')}`);
        return;
      }

      log('==================== ALL STEPS COMPLETED ====================');
      log('✓ Form automation completed successfully!');
      
      // Alert user
      alert('✓ Form automation completed successfully!\n\nCheck browser console (F12) for detailed audit log.');

    } catch (error) {
      logError('FORM AUTOMATION FAILED', error);
      console.error('Full automation failed:', error);
      console.error('Audit log:', AUDIT_LOG);
      alert(`❌ Form automation encountered an error:\n\n${error.message}\n\nCheck browser console (F12) for details.`);
    }
  }

  // ============================================================================
  // INITIALIZATION (MANUAL-ONLY MODE)
  // ============================================================================

  log('Automation script loaded (manual-only mode)');
  console.error('[PF-AUTO] Userscript injected on:', window.location.href);

  let triggerBtn = null;

  function createTriggerButton() {
    if (triggerBtn) return;

    triggerBtn = document.createElement('button');
    triggerBtn.id = 'ehr-automation-trigger';
    triggerBtn.textContent = 'Run EHR Auto-Fill';
    triggerBtn.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      z-index: 10000;
      padding: 12px 18px;
      background-color: #4CAF50;
      color: white;
      border: none;
      border-radius: 5px;
      cursor: pointer;
      font-size: 13px;
      font-weight: bold;
      box-shadow: 0 3px 8px rgba(0,0,0,0.3);
      font-family: Arial, sans-serif;
    `;
    triggerBtn.onmouseover = () => triggerBtn.style.backgroundColor = '#45a049';
    triggerBtn.onmouseout = () => triggerBtn.style.backgroundColor = '#4CAF50';
    triggerBtn.onclick = executeFormAutomation;
    document.body.appendChild(triggerBtn);
    log('Manual trigger button added (patient page)');
  }

  function removeTriggerButton() {
    if (!triggerBtn) return;
    triggerBtn.remove();
    triggerBtn = null;
    log('Manual trigger button removed (left patient page)');
  }

  function refreshUiForRoute() {
    if (isOnTargetChartPage()) {
      createTriggerButton();
      return;
    }
    removeTriggerButton();
  }

  function watchSpaRouteChanges() {
    const pushState = history.pushState;
    const replaceState = history.replaceState;

    history.pushState = function() {
      const result = pushState.apply(this, arguments);
      hasExecutedForThisPage = false;
      setTimeout(refreshUiForRoute, 500);
      return result;
    };

    history.replaceState = function() {
      const result = replaceState.apply(this, arguments);
      hasExecutedForThisPage = false;
      setTimeout(refreshUiForRoute, 500);
      return result;
    };

    window.addEventListener('hashchange', () => {
      hasExecutedForThisPage = false;
      setTimeout(refreshUiForRoute, 500);
    });

    window.addEventListener('popstate', () => {
      hasExecutedForThisPage = false;
      setTimeout(refreshUiForRoute, 500);
    });
  }

  watchSpaRouteChanges();

  // Run once now and whenever route changes. No automatic form submission.
  setTimeout(refreshUiForRoute, 1200);

})();
