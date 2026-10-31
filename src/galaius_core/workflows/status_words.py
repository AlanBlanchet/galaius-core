"""The status words a client shows, ONE table: every word's key (`StatusWord`) and its words in each language — its label,
what it means, what to do next, and, for a word said about a number of things, its singular and plural (« 1 connecté »,
« 3 connectés »). A run's words agree with « une exécution » (« Terminée », « Arrêtée », « Interrompue »); a person (an agent)
has its own (« Travaille », « Attend », « Terminé »). A server writing a status in a message (a push notification) and a client drawing it read the same words;
a client generates its own table from this one."""

from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict

StatusWord = Literal["working", "at-work", "working-again", "pc-away", "silent", "resuming", "sign-in", "limit-paused",
                     "limit-held", "limit-over", "cap-held", "cap-cut", "on-agents", "waiting", "awaiting", "needs-you",
                     "done", "finished", "failed", "stopped", "lost", "relaunched", "retrying", "skipped", "not-run", "pinned",
                     "idle", "to-launch", "never-run", "ready", "low", "nothing-owed", "cap-reached", "empty", "paused",
                     "checking", "online", "runs-nothing", "not-answering", "offline", "connecting", "not-connected-yet",
                     "not-connecting", "not-connected", "signed-out", "agents-off", "no-programs", "outdated", "no-folder",
                     "connected", "reconnect", "error", "not-tested", "needs-access", "active", "trial", "payment-due", "held",
                     "cancelled", "suspended", "not-subscribed", "verified", "not-verified", "saved", "revoked", "linked",
                     "on", "off", "to-install", "on-pc", "cannot", "installed", "unknown"]

#: English, then French.
Said = tuple[str, str]


class StatusPlural(BaseModel):
    """A word said about a number of things, lower case as it follows the number: (one, many) per language."""

    model_config = ConfigDict(frozen=True)

    en: Said
    fr: Said


class StatusText(BaseModel):
    """A status word's words: its label, its one-line meaning, what to do next (None: nothing to do), its plural."""

    model_config = ConfigDict(frozen=True)

    label: Said
    meaning: Said
    next: Said | None
    plural: StatusPlural | None = None

    def say(self, locale: str) -> str:
        """Its label in `locale`: English for "en", French for every other (the app's own default)."""
        return self.label[0 if locale == "en" else 1]


STATUS_TEXT: dict[StatusWord, StatusText] = {
    "working": StatusText(label=("Working", "En cours"), meaning=("Running right now.", "S’exécute en ce moment."), next=("Open it to watch.", "Ouvrez-le pour suivre.")),
    "at-work": StatusText(label=("Working", "Travaille"), meaning=("This agent is working right now.", "Cet agent travaille en ce moment."), next=("Open it to watch.", "Ouvrez-le pour suivre.")),
    "working-again": StatusText(label=("Working again", "Reprise · en cours"), meaning=("It had ended and runs again.", "Elle s’était arrêtée et tourne de nouveau."), next=("Open it to watch.", "Ouvrez-le pour suivre.")),
    "pc-away": StatusText(label=("PC offline", "PC hors ligne"), meaning=("Its PC stopped answering: it may still run there, but nothing reaches it until the PC is back. Not counted as working.", "Son PC ne répond plus : elle tourne peut-être encore, mais rien ne lui parvient avant son retour. Elle n’est pas comptée en cours."), next=("Turn the PC on: Galaius reconnects there by itself.", "Allumez le PC : Galaius s’y reconnecte tout seul.")),
    "silent": StatusText(label=("No news", "Sans nouvelles"), meaning=("Its computer still runs it, but it has written nothing for a while: it may be stuck.", "Son ordinateur l’exécute encore, mais elle n’a rien écrit depuis un moment : elle est peut-être bloquée."), next=("Open it to check, or stop it.", "Ouvrez-la pour vérifier, ou arrêtez-la.")),
    "resuming": StatusText(label=("Resumes by itself", "Reprise automatique"), meaning=("It was cut off; its PC resumes it by itself.", "Elle a été coupée ; son PC la reprend tout seul."), next=("Nothing to do.", "Rien à faire.")),
    "sign-in": StatusText(label=("Waits for a sign-in", "Attend la reconnexion"), meaning=("Its program's login expired on its PC: it resumes by itself once signed in again.", "La connexion de son programme a expiré sur son PC : elle reprend toute seule dès la reconnexion."), next=("Sign its program in again.", "Reconnectez son programme.")),
    "limit-paused": StatusText(label=("Paused", "En pause"), meaning=("Its provider's usage limit was reached. Its PC resumes it by itself when the limit resets.", "La limite d’usage de son fournisseur est atteinte. Son PC la reprend tout seul quand la limite se réinitialise."), next=("Nothing to do: it resumes by itself.", "Rien à faire : elle reprend toute seule.")),
    "limit-held": StatusText(label=("Blocked", "Bloquée"), meaning=("Its provider's usage limit stopped it and still holds.", "La limite d’usage de son fournisseur l’a arrêtée et tient encore."), next=("Wait for the reset, then write to it to pick it up.", "Attendez la réinitialisation, puis écrivez-lui pour la reprendre.")),
    "limit-over": StatusText(label=("Ready to continue", "Prête à continuer"), meaning=("Its provider's usage limit stopped it. That limit is over now.", "La limite d’usage de son fournisseur l’a arrêtée. Cette limite est passée."), next=("Nothing to do: it resumes by itself.", "Rien à faire : elle reprend toute seule.")),
    "cap-held": StatusText(label=("Held at its cap", "En pause au plafond"), meaning=("Its subscription reached the cap you set: it finished its turn and starts no other until the window reopens.", "Son abonnement a atteint le plafond que vous avez fixé : elle a fini son tour et n’en commence pas d’autre avant que la fenêtre se rouvre."), next=("Allow more for this task, or wait for the reset.", "Autorisez plus pour cette tâche, ou attendez la reprise.")),
    "cap-cut": StatusText(label=("Cut at its cap", "Coupée au plafond"), meaning=("Its subscription reached the cap you set: it was stopped at that reading.", "Son abonnement a atteint le plafond que vous avez fixé : elle a été arrêtée à ce relevé."), next=("Allow more for this task, or wait for the reset.", "Autorisez plus pour cette tâche, ou attendez la reprise.")),
    "on-agents": StatusText(label=("Waiting on its agents", "Attend ses agents"), meaning=("Its own turn is over; agents it launched still work.", "Son tour est fini ; des agents qu’il a lancés travaillent encore."), next=("Open it to see which.", "Ouvrez-le pour voir lesquels.")),
    "waiting": StatusText(label=("Waiting", "En attente"), meaning=("Paused until it gets what it needs to go on.", "En pause jusqu’à ce qu’elle obtienne ce qu’il lui faut pour continuer."), next=("Open it to see what it waits for.", "Ouvrez-le pour voir ce qu’il attend.")),
    "awaiting": StatusText(label=("Waiting", "Attend"), meaning=("This agent waits for what it needs to go on.", "Cet agent attend ce qu’il lui faut pour continuer."), next=("Open it to see what it waits for.", "Ouvrez-le pour voir ce qu’il attend.")),
    "needs-you": StatusText(label=("Waiting for you", "Vous attend"), meaning=("It needs your answer: a question or an approval.", "Il lui faut votre réponse : une question ou une validation."), next=("Open it to answer.", "Ouvrez-le pour répondre.")),
    "done": StatusText(label=("Done", "Terminée"), meaning=("Finished without error.", "Terminée sans erreur."), next=("Open it to read the result.", "Ouvrez-le pour lire le résultat.")),
    "finished": StatusText(label=("Done", "Terminé"), meaning=("This agent's work is done.", "Le travail de cet agent est terminé."), next=("Open it to read the result.", "Ouvrez-le pour lire le résultat.")),
    "failed": StatusText(label=("Failed", "Échec"), meaning=("Did not work. Nothing else was affected.", "N’a pas fonctionné. Rien d’autre n’est touché."), next=("Open it to see why.", "Ouvrez-le pour voir pourquoi.")),
    "stopped": StatusText(label=("Stopped", "Arrêtée"), meaning=("It was stopped before the end.", "Elle a été arrêtée avant la fin."), next=("Write to it to pick it up where it stopped.", "Écrivez-lui pour la reprendre où elle s’est arrêtée.")),
    "lost": StatusText(label=("Interrupted", "Interrompue"), meaning=("It was cut off before it finished.", "Elle a été interrompue avant la fin."), next=("Write to it to pick it up where it stopped.", "Écrivez-lui pour la reprendre où elle s’est arrêtée.")),
    "relaunched": StatusText(label=("Relaunched", "Relancée"), meaning=("Its provider refused to start it; another run took its task.", "Son fournisseur a refusé de la lancer ; une autre exécution a repris sa tâche."), next=None),
    "retrying": StatusText(label=("Retrying", "Nouvel essai"), meaning=("Its last try failed; it is trying again.", "Son dernier essai a échoué ; il réessaie."), next=None),
    "skipped": StatusText(label=("Skipped", "Ignoré"), meaning=("The run went on without this step.", "L’exécution a continué sans cette étape."), next=None),
    "not-run": StatusText(label=("Not run", "Pas exécuté"), meaning=("The last run did not reach this step.", "La dernière exécution n’a pas atteint cette étape."), next=None),
    "pinned": StatusText(label=("Pinned", "Épinglé"), meaning=("Its output is pinned: runs reuse it instead of running it.", "Sa sortie est épinglée : les exécutions la réutilisent au lieu de la relancer."), next=None),
    "idle": StatusText(label=("Idle", "Au repos"), meaning=("Nothing running now.", "Rien ne tourne en ce moment."), next=None),
    "to-launch": StatusText(label=("To launch", "À lancer"), meaning=("Ready: nothing started yet.", "Prête : rien n’est encore lancé."), next=("Press Launch.", "Appuyez sur Lancer.")),
    "never-run": StatusText(label=("Never run", "Jamais lancé"), meaning=("Saved, but never run yet.", "Enregistré, jamais lancé."), next=("Open it and press Run.", "Ouvrez-le et lancez-le.")),
    "ready": StatusText(label=("Ready", "Prêt"), meaning=("Set up and working.", "Configuré et fonctionnel."), next=None),
    "low": StatusText(label=("Low", "Bas"), meaning=("Credits will run out soon.", "Les crédits seront bientôt épuisés."), next=("Add credits, or turn on auto top-up.", "Ajoutez des crédits ou activez le rechargement automatique.")),
    "nothing-owed": StatusText(label=("Nothing to pay", "Rien à payer"), meaning=("No prepaid credits and nothing owed. Agents on your PCs' subscription run as usual; only a paid API call needs credits.", "Aucun crédit prépayé et rien de dû. Les agents sur l’abonnement de vos PC tournent comme d’habitude ; seul un appel d’API payant demande des crédits."), next=("Add credits only for paid API use.", "Ajoutez des crédits seulement pour l’usage payant de l’API.")),
    "cap-reached": StatusText(label=("Hard limit reached", "Limite stricte atteinte"), meaning=("This month's API spending reached the organisation's hard limit: every paid call is refused until next month or a higher limit. Your subscriptions keep running.", "La dépense API du mois a atteint la limite stricte de l’organisation : tout appel payant est refusé jusqu’au mois prochain ou à une limite plus haute. Vos abonnements continuent."), next=("Raise the limit below, or wait for next month.", "Relevez la limite ci-dessous, ou attendez le mois prochain.")),
    "empty": StatusText(label=("Empty", "Vide"), meaning=("No credits left. Paid runs are refused unless auto top-up can charge your card.", "Plus de crédits. Les exécutions payantes sont refusées, sauf si le rechargement automatique peut débiter votre carte."), next=("Add credits.", "Ajoutez des crédits.")),
    "paused": StatusText(label=("Paused", "En pause"), meaning=("Paid runs are stopped until credits are added.", "Les exécutions payantes sont arrêtées jusqu’à l’ajout de crédits."), next=("Add credits or update your card.", "Ajoutez des crédits ou changez de carte.")),
    "checking": StatusText(label=("Checking", "Vérification"), meaning=("Being checked right now.", "Vérification en cours."), next=("Wait a moment.", "Patientez un instant.")),
    "online": StatusText(label=("Connected", "Connecté"), meaning=("Connected now.", "Connecté en ce moment."), next=None, plural=StatusPlural(en=("connected", "connected"), fr=("connecté", "connectés"))),
    "runs-nothing": StatusText(label=("Connected, runs nothing", "Connecté, n’exécute rien"), meaning=("Connected, but it refuses what Galaius asks of it: it runs no agent until that is fixed.", "Connecté, mais il refuse ce que Galaius lui demande : il n’exécute aucun agent avant que ce soit réglé."), next=("Its reason and what to do are on that PC’s own page, in red.", "Son motif et quoi faire sont sur la page de ce PC, en rouge."), plural=StatusPlural(en=("connected, runs nothing", "connected, run nothing"), fr=("connecté, n’exécute rien", "connectés, n’exécutent rien"))),
    "not-answering": StatusText(label=("Not answering", "Ne répond pas"), meaning=("Connected, but it answers none of what it is sent: its messages wait.", "Connecté, mais il ne répond à rien de ce qu’on lui envoie : ses messages attendent."), next=("Open its page, or pick another PC.", "Ouvrez sa page, ou choisissez un autre PC."), plural=StatusPlural(en=("not answering", "not answering"), fr=("ne répond pas", "ne répondent pas"))),
    "offline": StatusText(label=("Offline", "Hors ligne"), meaning=("The PC is not connected to Galaius. It runs nothing until it is back.", "Le PC n’est pas relié à Galaius en ce moment. Il n’exécute rien jusqu’à son retour."), next=("Turn it on; Galaius reconnects by itself.", "Allumez-le ; Galaius se reconnecte tout seul."), plural=StatusPlural(en=("offline", "offline"), fr=("hors ligne", "hors ligne"))),
    "connecting": StatusText(label=("Connecting", "Connexion en cours"), meaning=("Just allowed: Galaius is starting on this PC and connecting.", "Tout juste autorisé : Galaius démarre sur ce PC et se connecte."), next=("Open its page to follow it.", "Ouvrez sa page pour suivre.")),
    "not-connected-yet": StatusText(label=("Not connected yet", "Pas encore connecté"), meaning=("Allowed, but it has never connected.", "Autorisé, mais il ne s’est encore jamais connecté."), next=("Open its page.", "Ouvrez sa page."), plural=StatusPlural(en=("not connected yet", "not connected yet"), fr=("pas encore connecté", "pas encore connectés"))),
    "not-connecting": StatusText(label=("Not connecting", "Ne se connecte pas"), meaning=("Galaius is installed on this PC but does not connect.", "Galaius est installé sur ce PC mais ne se connecte pas."), next=("Open its page.", "Ouvrez sa page."), plural=StatusPlural(en=("not connecting", "not connecting"), fr=("ne se connecte pas", "ne se connectent pas"))),
    "not-connected": StatusText(label=("Not connected", "Non connecté"), meaning=("Nothing is set up here yet. Nothing can run through it.", "Rien n’est configuré ici. Rien ne peut passer par là."), next=("Open it to connect.", "Ouvrez-le pour connecter.")),
    "signed-out": StatusText(label=("Signed out", "Déconnecté"), meaning=("Its agent program is signed out on this PC.", "Son programme d’agent est déconnecté sur ce PC."), next=("Sign it in again on the PC.", "Reconnectez-le sur le PC.")),
    "agents-off": StatusText(label=("Agents off", "Agents arrêtés"), meaning=("Agents are switched off on this PC.", "Les agents sont désactivés sur ce PC."), next=("Turn them on from its page.", "Activez-les depuis sa page.")),
    "no-programs": StatusText(label=("No agent program", "Aucun programme"), meaning=("No agent program is installed on this PC.", "Aucun programme d’agent n’est installé sur ce PC."), next=("Install one from its page.", "Installez-en un depuis sa page.")),
    "outdated": StatusText(label=("To update", "À mettre à jour"), meaning=("Galaius on this PC is too old for this.", "Galaius est trop ancien sur ce PC pour ceci."), next=("Update it from its page.", "Mettez-le à jour depuis sa page.")),
    "no-folder": StatusText(label=("No folder", "Pas de dossier"), meaning=("The project has no folder on this PC.", "Le projet n’a pas de dossier sur ce PC."), next=("Choose its folder there.", "Choisissez son dossier sur ce PC.")),
    "connected": StatusText(label=("Connected", "Connecté"), meaning=("Agents you allow read it, live, where it lives.", "Les agents que vous autorisez le lisent en direct, là où c’est rangé."), next=None),
    "reconnect": StatusText(label=("Reconnect", "À reconnecter"), meaning=("The provider no longer accepts the access you gave: nothing is read until you reconnect.", "Le fournisseur n’accepte plus l’accès donné : rien n’est lu tant que vous ne reconnectez pas."), next=("Press Reconnect.", "Appuyez sur Reconnecter.")),
    "error": StatusText(label=("Error", "Erreur"), meaning=("The last read failed.", "La dernière lecture a échoué."), next=("Read the reason, then try again.", "Lisez la raison, puis réessayez.")),
    "not-tested": StatusText(label=("Not tested yet", "Pas encore testée"), meaning=("Saved, but no check has reached it yet: it may or may not work.", "Enregistrée, mais aucune vérification ne l’a encore atteinte : elle peut marcher ou non."), next=("Press Test now.", "Appuyez sur Tester maintenant.")),
    "needs-access": StatusText(label=("Needs access", "Accès requis"), meaning=("Connected, but a permission is missing.", "Connecté, mais une autorisation manque."), next=("Open it and grant the access it lists.", "Ouvrez-le et accordez l’accès indiqué.")),
    "active": StatusText(label=("Active", "Actif"), meaning=("Paid and running.", "Payé et en cours."), next=None),
    "trial": StatusText(label=("Trial", "Essai"), meaning=("Free trial; billing starts when it ends.", "Essai gratuit ; la facturation commence à la fin."), next=("Add a card before it ends.", "Ajoutez une carte avant la fin.")),
    "payment-due": StatusText(label=("Payment due", "Paiement dû"), meaning=("The last payment did not go through.", "Le dernier paiement n’est pas passé."), next=("Update your card.", "Mettez à jour votre carte.")),
    "held": StatusText(label=("Held", "Retenue"), meaning=("Reserved; it starts at its hour.", "Réservée ; elle démarre à son heure."), next=None),
    "cancelled": StatusText(label=("Cancelled", "Annulé"), meaning=("Ended; it no longer renews.", "Terminé ; ne se renouvelle plus."), next=("Subscribe again if you need it.", "Réabonnez-vous si besoin.")),
    "suspended": StatusText(label=("Suspended", "Suspendu"), meaning=("Switched off for now; nothing runs on it until it is reactivated.", "Désactivé pour l’instant ; rien n’y tourne avant sa réactivation."), next=None),
    "not-subscribed": StatusText(label=("Not subscribed", "Non abonné"), meaning=("No plan yet.", "Pas encore de formule."), next=("Choose the plan below.", "Choisissez la formule ci-dessous.")),
    "verified": StatusText(label=("Verified", "Vérifié"), meaning=("Checked with the official register.", "Vérifié auprès du registre officiel."), next=None),
    "not-verified": StatusText(label=("Not verified", "Non vérifié"), meaning=("Not yet checked with the official register.", "Pas encore vérifié auprès du registre officiel."), next=("Check the number, then save again.", "Vérifiez le numéro, puis enregistrez.")),
    "saved": StatusText(label=("Saved", "Enregistré"), meaning=("Your latest changes are stored on the server.", "Vos dernières modifications sont stockées sur le serveur."), next=None),
    "revoked": StatusText(label=("Revoked", "Révoquée"), meaning=("Switched off for good. Anything still using it is refused.", "Désactivée pour de bon. Tout ce qui l’utilise encore est refusé."), next=("Create a new key if a script still needs access.", "Créez une nouvelle clé si un script a encore besoin d’accès.")),
    "linked": StatusText(label=("Linked", "Lié"), meaning=("Signed in here. What it may read is chosen per product.", "Connecté ici. Ce qu’il peut lire se choisit par produit."), next=None),
    "on": StatusText(label=("On", "Activé"), meaning=("Agents you allow read it.", "Les agents que vous autorisez le lisent."), next=None),
    "off": StatusText(label=("Off", "Désactivé"), meaning=("Agents read nothing here.", "Les agents n’y lisent rien."), next=("Turn it on.", "Activez-le.")),
    "to-install": StatusText(label=("To install", "À installer"), meaning=("Missing now; « Install » does it from here.", "Manque pour l’instant ; « Installer » s’en charge d’ici."), next=("Press Install.", "Appuyez sur Installer.")),
    "on-pc": StatusText(label=("On the PC", "Sur le PC"), meaning=("Only the PC's owner can do this, on the PC itself.", "Seul le propriétaire du PC peut le faire, sur le PC lui-même."), next=("Run the command shown on that PC.", "Lancez la commande indiquée sur ce PC.")),
    "cannot": StatusText(label=("Not possible", "Impossible"), meaning=("This PC cannot take this step as the workflow has it.", "Ce PC ne peut pas prendre cette étape telle que le workflow la décrit."), next=("Pick another PC, or change the step.", "Choisissez un autre PC, ou modifiez l’étape.")),
    "installed": StatusText(label=("Installed", "Installé"), meaning=("The workflow's PC steps run on this PC.", "Les étapes sur PC du workflow tournent sur ce PC."), next=None),
    "unknown": StatusText(label=("Unknown", "Inconnu"), meaning=("The server did not answer for this just now.", "Le serveur n’a pas répondu pour ceci."), next=("Reload the page in a moment.", "Rechargez la page dans un instant.")),
}
if tuple(STATUS_TEXT) != get_args(StatusWord):
    raise RuntimeError("STATUS_TEXT must hold every StatusWord, in its order, and no other word")

#: A notification's status (the bell's row, a push): how its run ended — or `waiting` on HIM, or `working` once more after it
#: had ended.
NotificationStatus = Literal["done", "failed", "stopped", "waiting", "working"]
#: Each as a word of the table: what the bell, a toast and a push all say of it.
NOTIFICATION_STATUS_WORD: dict[NotificationStatus, StatusWord] = {"done": "done", "failed": "failed", "stopped": "stopped", "waiting": "needs-you",
                                                                  "working": "working-again"}

__all__ = ["NOTIFICATION_STATUS_WORD", "STATUS_TEXT", "NotificationStatus", "Said", "StatusPlural", "StatusText", "StatusWord"]
