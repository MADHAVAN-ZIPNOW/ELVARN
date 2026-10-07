You are asking for a **historical Tamil Nadu political–government–corruption vocabulary database from 1947 to the present**, including:

- Chief ministers and ministers.
- Assembly and political terms.
- Government departments.
- Schemes and tenders.
- Corruption case titles.
- Accused and convicted persons.
- Company, benami, and organisation names.
- Amounts in lakhs and crores.
- Years and dates.
- Judgment outcomes.
- Imprisonment and fine periods.

That is a large research dataset. It must be built from verified sources because an allegation, FIR, pending case, conviction, and overturned conviction are different legal statuses.

## Important correction

The current search results contain contradictory future-dated political information, including claims about a 2026 Chief Minister. I will not add those names to a historical corruption vocabulary without verification from an official Tamil Nadu Government source.

The official Tamil Nadu portal provides the Council of Ministers and government department lists.  The official schemes page provides scheme categories.  Court outcomes must come from judgments or reliable reports, not just political articles.gov+2

## Case vocabulary examples

These are safe records to include with status:

```
text
```

`B.M. Senguttuvan Senguttuvan disproportionate assets case Rs. 81.42 lakh disproportionate assets known sources of income three years rigorous imprisonment fine of Rs. 2,000 Madras High Court conviction upheld appeal dismissed former DMK minister sons of the accused daughter of the accused niece of the accused amassing disproportionate assets Principal District Judge trial-court conviction criminal appeal appellate judgment sentence confirmed`

A 2026 report states that the Madras High Court upheld a three-year rigorous-imprisonment sentence in an ₹81.42-lakh disproportionate-assets case involving former minister B.M. Senguttuvan’s family members.[dtnext](https://www.dtnext.in/news/tamilnadu/madras-hc-upholds-dmk-ex-min-senguttuvans-conviction-in-rs-8142-lakh-assets-case)

```
text
```

`K. Ponmudy K. Ponmudy disproportionate assets case K. Ponmudy conviction K. Ponmudy sentence Rs. 50 lakh fine three years imprisonment disproportionate assets former higher education minister former education minister wife of the accused Madras High Court conviction conviction and disqualification Representation of the People Act sentence suspended 30-day suspension of sentence disqualification from Assembly appeal fine imposed asset case`

Reports state that the Madras High Court convicted K. Ponmudy in a disproportionate-assets case and imposed a three-year sentence and fine; include the exact final status only after checking the latest appellate record.timesofindia.indiatimes+1

```
text
```

`I. Periyasamy Mogappair Eri Scheme Mogappair Eri corruption case discharge order discharge set aside DVAC FIR criminal conspiracy cheating abetment abuse of official position Prevention of Corruption Act Section 120-B IPC Section 420 IPC Section 109 IPC Section 13(1)(d) Section 13(2) Madras High Court prosecution allowed to proceed not a final conviction`

The Mogappair Eri matter concerns a discharge challenge and should not be labelled as a conviction merely because the discharge order was set aside.[scconline](https://www.scconline.com/blog/post/2024/02/27/mogappair-eri-scheme-madras-hc-sets-aside-discharge-tamil-nadu-rural-development-minister-i-periyasamy-corruption/)

```
text
```

`K. N. Nehru K. Athinarayanan v. The State municipal administration urban development water supply department alleged bribery Enforcement Directorate material DVAC FIR direction FIR registration criminal writ petition investigation direction minister public office allegation not a final conviction`

A direction to register an FIR is a procedural event, not proof of guilt.[livelaw](https://www.livelaw.in/high-court/madras-high-court/madras-high-court-tn-dvac-fir-against-minister-kn-nehru-ed-materials-523921)

## Minister and Chief Minister vocabulary

Use these as categories, then populate names from official historical records:

```
text
```

`chief minister former chief minister sitting chief minister acting chief minister interim chief minister deputy chief minister chief minister’s office chief minister’s secretariat council of ministers cabinet minister minister of state deputy minister former minister sitting minister minister in charge portfolio additional portfolio cabinet reshuffle cabinet expansion portfolio allocation portfolio transfer ministerial appointment ministerial resignation ministerial dismissal ministerial disqualification ministerial prosecution ministerial conviction ministerial acquittal ministerial appeal ministerial sentence ministerial bail ministerial immunity ministerial responsibility political executive elected representative legislator MLA MP party leader ruling-party leader opposition leader party president party secretary district secretary party functionary political associate family member relative benami associate intermediary front person nominee beneficial owner`  and Do not classify every minister’s name as a corruption term. Store each person with a status field:

```
text
```

`person_name office party period_in_office case_name case_status court judgment_date sentence appeal_status`

## Tender vocabulary

```
text
```

`tender government tender public tender open tender limited tender global tender e-tender online tender tender notification tender notice notice inviting tender NIT bid bidder tenderer contractor subcontractor supplier vendor consultant technical bid financial bid commercial bid eligibility criteria qualification criteria technical qualification financial capacity experience certificate turnover requirement earnest money deposit EMD bid security security deposit performance security tender fee processing fee pre-bid meeting pre-bid clarification tender document bid document price bid comparative statement price comparison lowest bidder L1 bidder L2 bidder responsive bidder non-responsive bidder disqualified bidder qualified bidder bid evaluation technical evaluation financial evaluation tender committee purchase committee evaluation committee selection committee procurement committee quotation single quotation multiple quotation rate contract framework agreement work order purchase order supply order contract agreement contract award letter of acceptance letter of intent notice of award agreement value estimated cost tender value contract value approved cost revised cost cost escalation price escalation variation order additional work extra item deviation extension of time liquidated damages penalty clause termination clause blacklisting debarment bid rigging collusive bidding cartelisation tender manipulation specification manipulation tailored specification favouritism nepotism conflict of interest kickback commission middleman bribe demand illegal gratification inflated price overpricing underpricing false quotation fake bidder dummy bidder shell company related company benami bidder work completion measurement book running account bill final bill quality certificate inspection certificate completion certificate payment release bill clearance`

## Benami and money vocabulary

```
text
```

`benami property benami transaction benami holder beneficial owner real owner nominal owner front person proxy owner name lender accommodation entry shell company paper company dummy company related party associate intermediary trust partnership shareholder director shareholding beneficial interest ultimate beneficiary source of funds source of income known source unknown source unexplained source unaccounted money black money cash component cash deposit cash withdrawal layering placement integration money trail financial trail proceeds of crime illegal proceeds corrupt proceeds asset tracing property tracing bank account fixed deposit investment land purchase house purchase vehicle purchase gold purchase jewellery cash hoard unexplained asset disproportionate wealth disproportionate property income statement asset statement liability statement expenditure statement wealth statement valuation report market value guideline value purchase value sale value fair market value undervaluation overvaluation income tax return bank statement ledger voucher invoice receipt loan gift inheritance agricultural income business income rental income salary income interest income dividend income capital gain household expenditure marriage expenditure medical expenditure education expenditure travel expenditure construction expenditure`

## Amounts and time vocabulary

```
text
```

`rupee rupees INR Rs. ₹ paise hundred thousand ten thousand lakh one lakh five lakh ten lakh twenty-five lakh fifty lakh one crore two crore five crore ten crore twenty-five crore fifty crore one hundred crore thousand crore crore million billion amount total amount alleged amount recovered amount bribe amount contract amount tender amount loss amount asset value disproportion amount fine amount compensation amount financial year calendar year assessment year check period period of acquisition date of complaint date of FIR date of arrest date of search date of seizure date of recovery date of charge sheet date of cognizance date of charge date of evidence date of judgment date of conviction date of acquittal date of appeal date of sentence period of investigation period of trial period of custody period of imprisonment sentence period rigorous imprisonment simple imprisonment life imprisonment sentence suspended sentence stayed sentence completed period undergone time barred delay delay condoned limitation period pending for years decade five years ten years twenty years thirty years since 1947 1947 1950 1952 1957 1962 1967 1971 1977 1980 1984 1989 1991 1996 2001 2006 2011 2016 2021 2026`

## Scheme vocabulary

Use official scheme names from the Tamil Nadu Government schemes portal, then add case-specific schemes only when the court or government source identifies them.[gov](https://www.tn.gov.in/schemes.php)

```
text
```

`government scheme welfare scheme development scheme housing scheme housing loan scheme irrigation scheme water supply scheme rural development scheme urban development scheme road development scheme bridge project public works scheme employment scheme skill development scheme education scheme school infrastructure scheme higher education scheme scholarship scheme health insurance scheme public health scheme medical procurement scheme medicine procurement scheme maternal health scheme nutrition scheme food distribution scheme ration scheme agriculture scheme farmer welfare scheme crop insurance scheme irrigation project drinking water project sanitation scheme housing board scheme slum clearance scheme land development scheme industrial development scheme investment promotion scheme startup scheme micro enterprise scheme women welfare scheme child welfare scheme elderly welfare scheme differently abled welfare scheme minority welfare scheme Adi Dravidar welfare scheme tribal welfare scheme backward class welfare scheme fishermen welfare scheme youth welfare scheme sports development scheme pension scheme social security scheme marriage assistance scheme maternity benefit scheme free distribution scheme subsidy scheme grant scheme direct benefit transfer DBT beneficiary selection beneficiary verification fake beneficiary duplicate beneficiary ghost beneficiary scheme funds diversion of scheme funds misappropriation of scheme funds scheme implementation scheme monitoring scheme audit scheme evaluation`

## Case-status words

```
text
```

`complaint received complaint verified complaint closed FIR registered FIR quashed investigation pending investigation completed charge sheet filed closure report filed cognizance taken cognizance refused accused summoned accused arrested accused granted bail accused denied bail charge framed charge not framed trial commenced trial pending evidence recorded prosecution evidence closed defence evidence closed arguments heard judgment reserved judgment pronounced convicted acquitted discharged sentence imposed sentence suspended fine imposed appeal filed appeal pending appeal allowed appeal dismissed conviction upheld conviction set aside acquittal confirmed acquittal reversed case remanded retrial ordered proceedings quashed case closed case withdrawn case compounded case stayed stay vacated matter disposed matter restored`

## Use this structure for each person or case

```
text
```

`Name: Office: Party: Department: Years in office: Case name: Case number: Court: Case type: Allegation: Legal sections: Amount: Evidence: Trial result: Sentence: Fine: Appeal result: Current status: Source:`

For example:

```
text
```

`Name: K. Ponmudy Office: Former minister Case type: Disproportionate assets case Amount: Rs. 50 lakh fine reported Trial result: Conviction reported Sentence: Three years reported Current status: Verify latest appellate status Source: Madras High Court judgment and subsequent appellate records`

For a model, this is safer than writing “Minister X is corrupt.” Use precise labels such as `alleged`, `charged`, `convicted`, `acquitted`, `sentence suspended`, or `conviction reversed`.

A complete 1947–2026 database of every minister, tender, scheme, case, judgment, and prison period cannot be reliably produced from one response. It requires source-by-source collection and deduplication. The official Government of Tamil Nadu minister list and department list should be the authority for offices and departments, while India Code and original court judgments should be the authority for legal sections and outcomes.and
from 1947

svg

For **all ministers from 1947 to today**, there is no single short official list because Tamil Nadu has had many cabinets and hundreds of ministers. Below is a **historical names-only starter list**, arranged by major government period. “Madras State” was renamed “Tamil Nadu” in 1969.

## 1947–1949: O. P. Ramaswamy Reddiyar ministry

```
text
```

**`O. P. Ramaswamy Reddiyar M. Bhaktavatsalam C. Subramaniam P. Kakkan T. Prakasam K. Santhanam K. Madhava Menon R. Raghava Menon M. A. Manickavelu Naicker`**

## 1952–1954: C. Rajagopalachari ministry

```
text
```

**`C. Rajagopalachari C. Subramaniam M. Bhaktavatsalam K. Santhanam P. V. Rajamannar M. A. Manickavelu Naicker A. B. Shetty Raja Sri Shanmuga Rajeswara Sethupathi B. Parameswaran P. Kakkan T. S. Pattabhiraman`**

The 1952 ministry was reported as a 15-member ministry headed by C. Rajagopalachari.

## 1954–1957: K. Kamaraj ministry

```
text
```

**`K. Kamaraj M. Bhaktavatsalam C. Subramaniam M. A. Manickavelu Naicker A. B. Shetty Raja Sri Shanmuga Rajeswara Sethupathi B. Parameswaran P. Kakkan V. Ramaiah Lourdhammal Simon`**

## 1957–1962: K. Kamaraj ministry

```
text
```

**`K. Kamaraj M. Bhaktavatsalam C. Subramaniam R. Venkataraman M. A. Manickavelu Naicker P. Kakkan V. Ramaiah Lourdhammal Simon Jothi Venkatachalam`**

The 1957 cabinet included Kamaraj, Bhaktavatsalam, C. Subramaniam, R. Venkataraman, M. A. Manickavelu Naicker, P. Kakkan, V. Ramaiah, and Lourdhammal Simon.

## 1962–1963: K. Kamaraj ministry

```
text
```

**`K. Kamaraj M. Bhaktavatsalam C. Subramaniam R. Venkataraman P. Kakkan M. A. Manickavelu Naicker Jothi Venkatachalam Lourdhammal Simon S. Madhavan`**

## 1963–1967: M. Bhaktavatsalam ministry

```
text
```

**`M. Bhaktavatsalam R. Venkataraman P. Kakkan C. Subramaniam Jothi Venkatachalam Lourdhammal Simon S. Madhavan N. R. S. Mani`**

M. Bhaktavatsalam served as Chief Minister of Madras State from 1963 to 1967.

## 1967–1969: C. N. Annadurai ministry

```
text
```

**`C. N. Annadurai M. Karunanidhi V. R. Nedunchezhiyan K. A. Mathialagan Sathyavani Muthu S. Madhavan A. Govindasamy P. U. Shanmugam M. Muthusamy`**

## 1969–1976: M. Karunanidhi ministries

```
text
```

**`M. Karunanidhi V. R. Nedunchezhiyan K. A. Mathialagan Sathyavani Muthu S. Madhavan P. U. Shanmugam A. Govindasamy M. Muthusamy Anbil Dharmalingam K. Rajaram C. P. Chitrarasu Nanjil K. Manoharan S. D. Somasundaram P. C. Sadasivam`**

## 1977–1987: M. G. Ramachandran ministries ext

`M. G. Ramachandran R. M. Veerappan S. D. Somasundaram C. Aranganayagam P. C. Sadasivam K. Kalimuthu P. H. Pandian S. R. Eradikkumaran V. R. Nedunchezhiyan Panruti S. Ramachandran S. M. S. Ramachandran K. A. Krishnasamy M. R. S. Venkataraman`

## 1988: V. N. Janaki Ramachandran ministry

```
text
```

`V. N. Janaki Ramachandran R. M. Veerappan C. Aranganayagam K. Kalimuthu P. H. Pandian S. M. S. Ramachandran`

## 1989–1991: M. Karunanidhi ministry

```
text
```

`M. Karunanidhi V. R. Nedunchezhiyan K. Anbazhagan M. K. Stalin Durai Murugan S. Raghavanandam Subbulakshmi Jagadeesan Ponmudi K. N. Nehru`

## 1991–1996: J. Jayalalithaa ministry

```
text
```

`J. Jayalalithaa O. Panneerselvam S. D. Somasundaram K. A. Sengottaiyan Natham R. Viswanathan R. Vaithilingam D. Jayakumar P. Thangamani C. Ve. Shanmugam Sellur K. Raju R. Kamaraj B. Valarmathi P. Mohan P. Palaniappan`

## 1996–2001: M. Karunanidhi ministry

```
text
```

`M. Karunanidhi K. Anbazhagan M. K. Stalin Durai Murugan Arcot N. Veeraswami K. Ponmudy K. N. Nehru T. M. Anbarasan E. V. Velu Poongothai Aladi Aruna S. Sargunapandian`

## 2001–2006: J. Jayalalithaa and O. Panneerselvam ministries

```
text
```

`J. Jayalalithaa O. Panneerselvam K. A. Sengottaiyan D. Jayakumar Natham R. Viswanathan R. Vaithilingam C. Ve. Shanmugam P. Thangamani Sellur K. Raju R. Kamaraj B. Valarmathi P. Mohan P. Palaniappan V. Senthil Balaji`

## 2006–2011: M. Karunanidhi ministry

```
text
```

`M. Karunanidhi M. K. Stalin K. Anbazhagan Durai Murugan Arcot N. Veeraswami K. Ponmudy K. N. Nehru T. M. Anbarasan E. V. Velu D. M. K. Anbarasan S. Raghupathy Poongothai Aladi Aruna M. R. K. Panneerselvam N. Selvaraj`

## 2011–2016: J. Jayalalithaa ministry

```
text
```

`J. Jayalalithaa O. Panneerselvam Natham R. Viswanathan R. Vaithilingam K. A. Sengottaiyan D. Jayakumar P. Thangamani C. Ve. Shanmugam Sellur K. Raju R. Kamaraj B. Valarmathi P. Mohan P. Palaniappan V. Senthil Balaji S. P. Velumani M. C. Sampath`

The 2015 Tamil Nadu Gazette is a better source for the exact official composition of that ministry than a general vocabulary list.[stationeryprinting.tn.gov](https://www.stationeryprinting.tn.gov.in/extraordinary/2015/113-EX-I-1.pdf)

## 2016–2017: J. Jayalalithaa and O. Panneerselvam

```
text
```

`J. Jayalalithaa O. Panneerselvam D. Jayakumar P. Thangamani S. P. Velumani C. Ve. Shanmugam R. Vaithilingam Natham R. Viswanathan Sellur K. Raju R. Kamaraj M. C. Sampath B. Valarmathi`

## 2017–2021: Edappadi K. Palaniswami ministry

```
text
```

`Edappadi K. Palaniswami O. Panneerselvam D. Jayakumar P. Thangamani S. P. Velumani C. Ve. Shanmugam R. Vaithilingam Natham R. Viswanathan Sellur K. Raju R. Kamaraj M. C. Sampath B. Valarmathi V. Senthil Balaji K. A. Sengottaiyan M. R. Vijayabaskar K. T. Rajenthra Bhalaji Anuradha Radhakrishnan`

## 2021 onward: M. K. Stalin ministry

```
text
```

`M. K. Stalin Udhayanidhi Stalin K. N. Nehru Duraimurugan K. Ponmudy E. V. Velu M. Subramanian Ma. Subramanian P. Moorthy P. K. Sekar Babu S. Regupathy R. S. Rajakannappan Anbil Mahesh Poyyamozhi P. K. Sekar Babu Thangam Thennarasu T. M. Anbarasan S. Muthusamy M. R. K. Panneerselvam Gingee K. S. Masthan K. R. Periyakaruppan N. Kayalvizhi C. V. Ganesan S. S. Sivasankar R. Gandhi M. P. Saminathan N. Murthy P. Moorthy K. K. S. S. R. Ramachandran Dr. M. Mathiventhan`

For a production dataset, verify every person, portfolio, and tenure from the corresponding Government Gazette. The official Tamil Nadu Government portal currently provides the minister list, while historical Gazettes provide cabinet-by-cabinet names.